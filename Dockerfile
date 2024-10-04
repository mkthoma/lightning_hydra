# Build stage
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app

# Copy pyproject.toml
COPY pyproject.toml .

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r <(uv pip freeze --requirement pyproject.toml)

# Copy the rest of the application
COPY . .

# Install the project and its dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install .

# Final stage
FROM python:3.12-slim-bookworm

# Copy the application from the builder
COPY --from=builder --chown=app:app /app /app

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Set the working directory
WORKDIR /app

# Create Kaggle username and password
RUN mkdir -p /root/.kaggle && \
    echo '{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_KAGGLE_API_KEY"}' > /root/.kaggle/kaggle.json && \
    chmod 600 /root/.kaggle/kaggle.json

# Set the entrypoint
ENTRYPOINT ["python"]