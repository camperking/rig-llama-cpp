use crate::error::LoadError;
use crate::types::{FitParams, KvCacheParams};

pub(crate) struct WorkerModel {
    pub(crate) model: llama_cpp_2::model::LlamaModel,
    #[cfg(feature = "mtmd")]
    pub(crate) mtmd_ctx: Option<llama_cpp_2::mtmd::MtmdContext>,
    pub(crate) n_ctx: u32,
    pub(crate) kv_cache: KvCacheParams,
}

/// Load a model with automatic parameter fitting to available device memory.
pub(crate) fn fit_and_load_model(
    backend: &llama_cpp_2::llama_backend::LlamaBackend,
    model_path: &str,
    mmproj_path: Option<&str>,
    n_ctx: u32,
    fit: &FitParams,
    kv_cache: &KvCacheParams,
    logs_enabled: bool,
) -> Result<WorkerModel, LoadError> {
    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::list_llama_ggml_backend_devices;
    use llama_cpp_2::model::LlamaModel as LlamaCppModel;
    use llama_cpp_2::model::params::LlamaModelParams;
    use std::num::NonZeroU32;
    use std::pin::pin;

    // Do NOT call with_n_gpu_layers — fit requires n_gpu_layers at default (-1)
    let mut model_params = LlamaModelParams::default();

    if backend.supports_gpu_offload() {
        let vulkan_devices: Vec<usize> = list_llama_ggml_backend_devices()
            .into_iter()
            .filter(|device| device.backend.eq_ignore_ascii_case("vulkan"))
            .map(|device| device.index)
            .collect();

        if !vulkan_devices.is_empty() {
            model_params = model_params
                .with_devices(&vulkan_devices)
                .map_err(|e| LoadError::ConfigureDevices(e.to_string()))?;
            log::info!("Using Vulkan backend devices: {vulkan_devices:?}");
        }
    }

    let mut pinned_params = pin!(model_params);

    // Context params for the fit call. `n_ctx` is left as the user's request;
    // `fit_params` only auto-selects when `n_ctx == 0`.
    let mut cparams = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(n_ctx));

    // Prepare margins
    let max_devices = unsafe { llama_cpp_sys_2::llama_max_devices() };
    let mut margins = fit
        .margins
        .clone()
        .unwrap_or_else(|| vec![1 << 30; max_devices]);
    margins.resize(max_devices, 1 << 30);

    let model_cstr =
        std::ffi::CString::new(model_path).map_err(|e| LoadError::InvalidPath(e.to_string()))?;

    // The C-side log level for `fit_params`. Routed via the
    // `RIG_LLAMA_CPP_LOGS` env var, not through the `log` facade, because
    // llama.cpp writes directly to stderr from C and bypasses Rust's logger.
    let log_level = if logs_enabled {
        llama_cpp_sys_2::GGML_LOG_LEVEL_INFO
    } else {
        llama_cpp_sys_2::GGML_LOG_LEVEL_NONE
    };

    log::info!("Fitting model parameters for {model_path}...");

    let fit_result = pinned_params
        .as_mut()
        .fit_params(
            &model_cstr,
            &mut cparams,
            &mut margins,
            fit.n_ctx_min,
            log_level,
        )
        .map_err(|e| LoadError::Fit(e.to_string()))?;

    let actual_n_ctx = fit_result.n_ctx;

    log::info!(
        "Fit complete: n_gpu_layers={}, n_ctx={}",
        pinned_params.n_gpu_layers(),
        actual_n_ctx
    );
    log::info!("Loading model from {model_path}...");

    let model = LlamaCppModel::load_from_file(backend, model_path, &pinned_params)
        .map_err(|e| LoadError::ModelLoad(e.to_string()))?;

    log::info!("Model loaded.");

    #[cfg(feature = "mtmd")]
    let mtmd_ctx = if let Some(mmproj) = mmproj_path {
        let mtmd_params = llama_cpp_2::mtmd::MtmdContextParams::default();
        let ctx = llama_cpp_2::mtmd::MtmdContext::init_from_file(mmproj, &model, &mtmd_params)
            .map_err(|e| LoadError::MmprojInit(e.to_string()))?;
        log::info!("Multimodal projector loaded from {mmproj}.");
        Some(ctx)
    } else {
        None
    };

    #[cfg(not(feature = "mtmd"))]
    let _ = mmproj_path;

    Ok(WorkerModel {
        model,
        #[cfg(feature = "mtmd")]
        mtmd_ctx,
        n_ctx: actual_n_ctx,
        kv_cache: *kv_cache,
    })
}
