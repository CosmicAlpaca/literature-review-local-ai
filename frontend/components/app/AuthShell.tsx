"use client";

import { useAccessToken } from "@/components/auth/queries";
import { Spinner } from "@/components/ui/spinner";
import { clearAccessToken } from "@/lib/auth";
import { useRouter, usePathname } from "next/navigation";
import { useEffect, useRef, type ReactNode } from "react";

type Props = {
  children: ReactNode;
};

const UNPROTECTED_ROUTES = ["/login", "/register"];
const PUBLIC_EXACT_ROUTES = ["/"];

export default function AuthShell({ children }: Props) {
  const router = useRouter();
  const pathname = usePathname() ?? "";
  const accessTokenQuery = useAccessToken();
  const redirectingRef = useRef(false);

  const isPublic = PUBLIC_EXACT_ROUTES.includes(pathname);
  const isAuthPage = UNPROTECTED_ROUTES.some((route) =>
    pathname.startsWith(route),
  );
  const isUnprotected = isAuthPage || isPublic;

  const hasToken = !accessTokenQuery.isPending && !!accessTokenQuery.data;
  const noToken =
    !accessTokenQuery.isPending &&
    (accessTokenQuery.isError || !accessTokenQuery.data);

  const shouldRedirectToLogin = noToken && !isUnprotected;
  const shouldRedirectToApp = hasToken && isAuthPage;

  useEffect(() => {
    redirectingRef.current = false;
  }, [pathname]);

  useEffect(() => {
    if (redirectingRef.current) return;

    if (shouldRedirectToLogin) {
      redirectingRef.current = true;
      clearAccessToken();
      router.replace("/login");
    } else if (shouldRedirectToApp) {
      redirectingRef.current = true;
      router.replace("/new-search");
    }
  }, [shouldRedirectToLogin, shouldRedirectToApp]);

  // On unprotected routes, render children immediately — no auth gate needed.
  if (isUnprotected) {
    return <>{children}</>;
  }

  if (
    accessTokenQuery.isPending ||
    shouldRedirectToLogin ||
    shouldRedirectToApp
  ) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <Spinner className="size-8 text-muted-foreground" />
      </div>
    );
  }

  return <>{children}</>;
}
