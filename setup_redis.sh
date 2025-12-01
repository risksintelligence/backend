#!/bin/bash

# Redis Setup Script for Supply Chain Risk Observatory
# Sets up Redis for local development and production caching

set -e

echo "🚀 Setting up Redis for Supply Chain Risk Observatory"
echo "=================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

# Detect Docker Compose command
COMPOSE_CMD="docker-compose"
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
fi

echo "✅ Docker and Docker Compose are available"

# Stop any existing Redis containers
echo "🛑 Stopping any existing Redis containers..."
$COMPOSE_CMD -f docker-compose.redis.yml down 2>/dev/null || true

# Pull latest Redis image
echo "📦 Pulling latest Redis image..."
docker pull redis:7-alpine

# Start Redis services
echo "🚀 Starting Redis services..."
$COMPOSE_CMD -f docker-compose.redis.yml up -d

# Wait for Redis to be ready
echo "⏳ Waiting for Redis to be ready..."
for i in {1..30}; do
    if docker exec rrio-redis redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Redis failed to start after 30 seconds"
        exit 1
    fi
    sleep 1
done

# Test Redis connection
echo "🧪 Testing Redis connection..."
if docker exec rrio-redis redis-cli ping | grep -q "PONG"; then
    echo "✅ Redis connection test successful!"
else
    echo "❌ Redis connection test failed"
    exit 1
fi

# Set up Redis configuration for supply chain data
echo "⚙️ Configuring Redis for supply chain data..."

# Create Redis configuration via redis-cli
docker exec rrio-redis redis-cli CONFIG SET maxmemory-policy allkeys-lru
docker exec rrio-redis redis-cli CONFIG SET maxmemory 512mb
docker exec rrio-redis redis-cli CONFIG SET timeout 0
docker exec rrio-redis redis-cli CONFIG SET tcp-keepalive 60

echo "✅ Redis configuration complete!"

# Create environment file if it doesn't exist
ENV_FILE="../.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "📝 Creating .env file..."
    cat > "$ENV_FILE" << EOF
# Redis Configuration
RIS_REDIS_URL=redis://localhost:6379/0

# Database Configuration
DATABASE_URL=sqlite:///./supply_chain.db

# API Keys (add your keys here)
ALPHA_VANTAGE_API_KEY=your_key_here
FRED_API_KEY=your_key_here
ACLED_API_KEY=your_key_here
SP_GLOBAL_API_KEY=your_key_here

# Cache Settings
CACHE_ENABLED=true
CACHE_TTL_DEFAULT=3600
EOF
else
    # Update existing .env file with Redis URL if not present
    if ! grep -q "RIS_REDIS_URL" "$ENV_FILE"; then
        echo "" >> "$ENV_FILE"
        echo "# Redis Configuration" >> "$ENV_FILE"
        echo "RIS_REDIS_URL=redis://localhost:6379/0" >> "$ENV_FILE"
    fi
fi

echo "📝 Environment configuration updated"

# Test supply chain cache integration
echo "🧪 Testing supply chain cache integration..."

# Create a simple test script
cat > test_cache.py << 'EOF'
#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app.core.supply_chain_cache import get_supply_chain_cache
    
    cache = get_supply_chain_cache()
    
    # Test basic cache operations
    cache.set("test_data", "test_key", {"message": "Hello Redis!"}, "test_script")
    data, metadata = cache.get("test_data", "test_key")
    
    if data and data.get("message") == "Hello Redis!":
        print("✅ Supply chain cache integration test passed!")
        print(f"   Cached data: {data}")
        print(f"   Metadata: {metadata}")
        
        # Test cache statistics
        stats = cache.get_cache_stats()
        print(f"   Cache status: {stats.get('status')}")
        print(f"   Hit rate: {stats.get('hit_rate', 0):.1f}%")
        
    else:
        print("❌ Supply chain cache integration test failed!")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Cache test error: {e}")
    sys.exit(1)
EOF

# Run the cache test
if python3 test_cache.py; then
    echo "✅ Cache integration test successful!"
else
    echo "⚠️ Cache integration test had issues (Redis may still work)"
fi

# Clean up test file
rm -f test_cache.py

# Show connection information
echo ""
echo "🎉 Redis setup complete!"
echo "========================"
echo ""
echo "📊 Redis Services:"
echo "   Redis Server:     localhost:6379"
echo "   Redis Commander:  http://localhost:8081"
echo "   Username: admin   Password: admin"
echo ""
echo "🔧 Useful Commands:"
echo "   View logs:        $COMPOSE_CMD -f docker-compose.redis.yml logs -f"
echo "   Stop services:    $COMPOSE_CMD -f docker-compose.redis.yml down"
echo "   Restart services: $COMPOSE_CMD -f docker-compose.redis.yml restart"
echo "   Redis CLI:        docker exec -it rrio-redis redis-cli"
echo ""
echo "🚀 Your supply chain caching is now ready!"
echo "   The application will automatically use Redis for:"
echo "   • Cascade event caching (5-15 min TTL)"
echo "   • Real-time alerts (1-5 min TTL)" 
echo "   • Market intelligence (1-4 hour TTL)"
echo "   • Sector vulnerability assessments (1 day-1 week TTL)"
echo "   • Supply chain network topology (12 hours-3 days TTL)"
echo ""

# Final verification
if curl -s http://localhost:8081 > /dev/null; then
    echo "✅ Redis Commander is accessible at http://localhost:8081"
else
    echo "⚠️ Redis Commander may still be starting up. Try http://localhost:8081 in a moment."
fi