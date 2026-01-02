# Use python 3.10
FROM python:3.10-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency definitions
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Copy the rest of the code
COPY . .

# Ensure the virtual environment is on the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Command to run (This runs your main script)
CMD ["uv", "run", "main.py"]
