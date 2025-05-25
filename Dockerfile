FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy everything including venv
COPY . .

# Activate venv and run the app
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Expose your Flask port
EXPOSE 5000

CMD ["python", "app.py"]
