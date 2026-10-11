# Dashboard image: build with Vite, serve the static bundle from nginx.
# nginx proxies /api to the API service, so the browser never talks to the
# backend origin directly.

FROM node:22-alpine AS build

WORKDIR /app

# Install deps first so this layer is cached across source-only changes.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build


FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1/ || exit 1
