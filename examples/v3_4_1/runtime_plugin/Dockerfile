FROM vllm/vllm-openai:v0.32.0

COPY examples/v3_4_1/runtime_plugin \
     /opt/user-vllm-ext

RUN python -m pip install --no-deps /opt/user-vllm-ext

RUN python -c \
    "from user_vllm_ext.scheduler import UserScheduler; print('V341_RUNTIME_IMPORT_PASS')"
