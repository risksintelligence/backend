# Frontend is now deployed separately as a static site
# This Dockerfile is kept for reference but frontend deployment
# now uses static site generation via render.yaml

FROM node:18-alpine AS base

# Install dependencies only when needed
FROM base AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app

# Copy package files from frontend directory
COPY ../../frontend/package.json ../../frontend/package-lock.json* ./
RUN npm ci && npm cache clean --force

# Build stage
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY ../../frontend/ ./

# Build and export static files
RUN npm run build
RUN npm run export

# Serve static files
FROM nginx:alpine AS runner
COPY --from=builder /app/out /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]