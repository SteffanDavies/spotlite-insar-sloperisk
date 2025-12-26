# Use python 3.10
FROM python:3.10-slim

# Install uv in the container
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency definition
COPY pyproject.toml .

# Install dependencies using uv
RUN uv sync --frozen

# Copy the rest of the code
COPY . .

# Command to run when container starts
CMD ["uv", "run", "main.py"]