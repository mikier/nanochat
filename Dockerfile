# syntax=docker/dockerfile:1
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

ARG APP_HOME=/app
WORKDIR ${APP_HOME}

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        curl \
        git \
        libssl-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain for building the rustbpe extension.
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal && \
    rustup component add rustfmt

COPY requirements.txt pyproject.toml uv.lock ./


RUN pip install uv

COPY . .

# Build and install the local package (compiles rustbpe through maturin).
RUN pip install --no-cache-dir --no-deps -e .

RUN chmod +x speedrun.sh

# Default run kicks off the speedrun pipeline; override CMD for other workflows.
CMD ["bash", "speedrun.sh"]

