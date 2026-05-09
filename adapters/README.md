# Adapters

Place your trained LoRA adapter files here before building the Docker image.

## Expected layout

```
adapters/
├── mobile/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── web/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

The service will fall back to the base model gracefully if these directories are empty.
