FROM node:22-slim AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
ARG NEXT_PUBLIC_API_BASE
ENV NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE
RUN npm run build
FROM node:22-slim
WORKDIR /app
ENV NODE_ENV=production
COPY --from=builder /app ./
CMD ["npm", "run", "start", "--", "--hostname", "0.0.0.0"]
