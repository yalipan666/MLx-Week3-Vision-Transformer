#!/bin/bash

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "🍺 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✅ Homebrew is already installed"
fi

# Update Homebrew
echo "🔄 Updating Homebrew..."
brew update
brew upgrade

# Install Docker if not installed
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    brew install --cask docker
    # Start Docker Desktop
    open -a Docker
    echo "⏳ Waiting for Docker to start..."
    sleep 30
else
    echo "✅ Docker is already installed"
fi

# Install Docker Compose if not installed
if ! command -v docker-compose &> /dev/null; then
    echo "🐳 Installing Docker Compose..."
    brew install docker-compose
else
    echo "✅ Docker Compose is already installed"
fi

# Create app directory if it doesn't exist
APP_DIR="$HOME/mnist-app"
if [ ! -d "$APP_DIR" ]; then
    echo "📁 Creating application directory..."
    mkdir -p "$APP_DIR"
fi

# Copy application files if they're not already in the directory
if [ ! -f "$APP_DIR/docker-compose.yml" ]; then
    echo "📋 Copying application files..."
    cp -r ./* "$APP_DIR/"
fi

# Change to app directory
cd "$APP_DIR"

# Build and start containers
echo "🚀 Building and starting containers..."
docker-compose up -d --build

# Check if containers are running
echo "🔍 Checking container status..."
docker-compose ps

# Show logs
echo "📝 Showing container logs..."
docker-compose logs -f 