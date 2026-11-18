import axios, {
  AxiosError,
  type AxiosRequestConfig,
  type Method,
  type RawAxiosRequestHeaders,
} from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

function getAccessToken() {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("access_token");
}

function withAuthHeaders(
  headers?: RawAxiosRequestHeaders | AxiosRequestConfig["headers"],
): AxiosRequestConfig["headers"] {
  const token = getAccessToken();
  const mergedHeaders: RawAxiosRequestHeaders = {
    ...(headers as RawAxiosRequestHeaders | undefined),
  };

  if (token) {
    mergedHeaders.Authorization = `Bearer ${token}`;
  }

  return mergedHeaders;
}

function readAxiosErrorMessage(error: AxiosError) {
  const data = error.response?.data;

  if (data && typeof data === "object") {
    const detail = (data as { detail?: unknown }).detail;
    if (typeof detail === "string" && detail.trim()) {
      return detail;
    }
  }

  if (typeof data === "string" && data.trim()) {
    return data;
  }

  return error.message || "API request failed";
}

/**
 * Request bodies we forward to axios as `data`.
 * Excludes `ReadableStream` (fetch allows it; axios/browser usage differs and is unused here).
 */
export type ApiFetchBody = Exclude<BodyInit, ReadableStream>;

export type ApiFetchOptions = Omit<RequestInit, "body"> & {
  body?: ApiFetchBody | null;
  /**
   * When true, HTTP 404 yields `null` instead of throwing (optional resource).
   */
  nullOn404?: boolean;
};

export type ApiFetchTextOptions = Omit<RequestInit, "body"> & {
  body?: ApiFetchBody | null;
};

function toAxiosConfig(path: string, options: ApiFetchTextOptions = {}): AxiosRequestConfig {
  const headers = withAuthHeaders(options.headers as AxiosRequestConfig["headers"]);
  const normalizedHeaders = headers as Record<string, string | undefined>;
  const hasContentType = Object.keys(normalizedHeaders).some(
    (key) => key.toLowerCase() === "content-type",
  );

  const { body } = options;

  // FormData / URLSearchParams: let axios set Content-Type (multipart boundary or urlencoded).
  // Blob: omit default so axios / blob.type can apply.
  const skipDefaultJsonContentType =
    body == null ||
    body instanceof FormData ||
    body instanceof URLSearchParams ||
    body instanceof Blob;

  if (!skipDefaultJsonContentType && !hasContentType) {
    normalizedHeaders["Content-Type"] = "application/json";
  }

  return {
    url: path,
    method: (options.method?.toLowerCase() as Method | undefined) ?? "get",
    headers: normalizedHeaders,
    data: body ?? undefined,
    signal: options.signal ?? undefined,
  };
}

const client = axios.create({
  baseURL: API_BASE,
});

export async function apiFetch<T>(
  path: string,
  options: ApiFetchOptions & { nullOn404: true },
): Promise<T | null>;
export async function apiFetch<T = unknown>(
  path: string,
  options?: ApiFetchOptions,
): Promise<T>;
export async function apiFetch<T>(
  path: string,
  options: ApiFetchOptions = {},
): Promise<T | null> {
  const { nullOn404, ...fetchInit } = options;

  try {
    const response = await client.request<T>({
      ...toAxiosConfig(path, fetchInit),
      validateStatus: (status) => {
        if (nullOn404 && status === 404) return true;
        return status >= 200 && status < 300;
      },
    });

    if (nullOn404 && response.status === 404) {
      return null;
    }

    if (response.status === 204) {
      return null;
    }

    return response.data as T;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(readAxiosErrorMessage(error));
    }

    throw new Error("API request failed");
  }
}

export async function apiFetchText(path: string, options: ApiFetchTextOptions = {}) {
  try {
    const response = await client.request<string>({
      ...toAxiosConfig(path, options),
      responseType: "text",
      validateStatus: (status) => status >= 200 && status < 300,
    });

    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      throw new Error(readAxiosErrorMessage(error));
    }

    throw new Error("API request failed");
  }
}
