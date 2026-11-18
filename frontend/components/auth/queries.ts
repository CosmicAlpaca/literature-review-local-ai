import { getAccessToken } from "@/lib/auth";
import { useQuery } from "@tanstack/react-query";

export const useAccessToken = () => {
  return useQuery({
    queryKey: ["accessToken"],
    queryFn: () => {
      const accessToken = getAccessToken();
      if (!accessToken) {
        return Promise.reject(new Error("No access token"));
      }
      return accessToken;
    },
    retry: false,
    staleTime: Infinity,
    refetchOnMount: false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  });
};
