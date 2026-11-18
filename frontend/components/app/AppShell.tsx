"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { type ReactNode, useEffect, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  PlusIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  LogOutIcon,
  BookOpenIcon,
  Trash2Icon,
  LayoutListIcon,
  FileTextIcon,
  FlaskConicalIcon,
} from "lucide-react";
import AppBrand, { APP_BRAND_NAME } from "@/components/app/AppBrand";
import { clearAccessToken } from "@/lib/auth";
import { apiFetch } from "@/lib/axios";
import {
  formatCreatedAt,
  getSessionTopicDisplay,
  normalizeRawQuery,
  useDeleteSession,
  useSessionList,
} from "@/lib/session";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
  PopoverHeader,
  PopoverTitle,
  PopoverDescription,
} from "@/components/ui/popover";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Separator } from "@/components/ui/separator";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";

type UserInfo = {
  id: number;
  email: string;
  full_name: string;
  role: string;
};

type Props = {
  children: ReactNode;
};

function getInitials(name: string) {
  return name
    .split(" ")
    .filter(Boolean)
    .slice(0, 2)
    .map((n) => n[0].toUpperCase())
    .join("");
}

export default function AppShell({ children }: Props) {
  const pathname = usePathname() ?? "";
  const router = useRouter();
  const [collapsed, setCollapsed] = useState(false);
  const queryClient = useQueryClient();

  const userInfoQuery = useQuery<UserInfo>({
    queryKey: ["userInfo"],
    queryFn: () => apiFetch("/auth/me"),
    retry: false,
  });

  // Shared session-list cache — see `useSessionList` for the full caching
  // contract. Sidebar, sessions index page, and session detail page all
  // subscribe to the same `queryKey`, so only one network call fires.
  const sessionsQuery = useSessionList({ enabled: userInfoQuery.isSuccess });

  const deleteMutation = useDeleteSession();
  // Chain navigation after the shared mutation already pruned the cache.
  const deleteSession = (sessionId: number) => {
    deleteMutation.mutate(sessionId, {
      onSuccess: () => {
        if (pathname === `/sessions/${sessionId}`) {
          router.replace("/sessions");
        }
      },
    });
  };

  // IMPORTANT: All hooks must run on every render in the same order. Keep
  // this `useMemo` (and any future hooks) above the conditional early
  // returns below, otherwise React throws "change in the order of Hooks".
  const sessions = useMemo(() => {
    const data = sessionsQuery.data;
    if (!data) return [];
    return [...data].sort(
      (a, b) =>
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
    );
  }, [sessionsQuery.data]);

  useEffect(() => {
    if (
      !userInfoQuery.isPending &&
      (userInfoQuery.isError || !userInfoQuery.data)
    ) {
      clearAccessToken();
      queryClient.clear();
      router.replace("/login");
    }
  }, [
    userInfoQuery.isPending,
    userInfoQuery.isError,
    userInfoQuery.data,
    router,
    queryClient,
  ]);

  if (userInfoQuery.isPending) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <Spinner className="size-8 text-muted-foreground" />
      </div>
    );
  }

  if (userInfoQuery.isError || !userInfoQuery.data) {
    return null;
  }

  const user = userInfoQuery.data;

  return (
    <TooltipProvider delayDuration={300}>
      <div className="flex h-screen bg-background overflow-hidden">
        {/* Sidebar */}
        <aside
          className={cn(
            "relative flex flex-col shrink-0 bg-card border-r border-border transition-all duration-200",
            collapsed ? "w-14" : "w-60",
          )}
        >
          {/* Toggle button */}
          <button
            type="button"
            onClick={() => setCollapsed((v) => !v)}
            className="absolute -right-3 top-6 z-10 flex h-6 w-6 items-center justify-center rounded-full border border-border bg-card text-muted-foreground shadow-sm hover:text-foreground transition-colors"
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? (
              <ChevronRightIcon className="h-3.5 w-3.5" />
            ) : (
              <ChevronLeftIcon className="h-3.5 w-3.5" />
            )}
          </button>

          {/* Logo / brand */}
          <div
            className={cn("px-3 py-4", collapsed && "flex justify-center")}
          >
            {collapsed ? (
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="inline-flex">
                    <AppBrand showName={false} size="sm" href="/sessions" />
                  </span>
                </TooltipTrigger>
                <TooltipContent side="right">{APP_BRAND_NAME}</TooltipContent>
              </Tooltip>
            ) : (
              <AppBrand size="sm" href="/sessions" />
            )}
          </div>

          <Separator />

          {/* New Search button */}
          <div className={cn("px-2 py-3", collapsed && "flex justify-center")}>
            {collapsed ? (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Link
                    href="/new-search"
                    className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
                    aria-label="New Search"
                  >
                    <PlusIcon className="h-4 w-4" />
                  </Link>
                </TooltipTrigger>
                <TooltipContent side="right">New Search</TooltipContent>
              </Tooltip>
            ) : (
              <Link
                href="/new-search"
                className="flex w-full items-center gap-2 rounded-lg bg-primary px-3 py-2 text-sm font-semibold text-primary-foreground hover:bg-primary/90 transition-colors"
              >
                <PlusIcon className="h-4 w-4 shrink-0" />
                New Search
              </Link>
            )}
          </div>

          {/* PDF Chat link */}
          <div className={cn("px-2 pb-2", collapsed && "flex justify-center")}>
            {collapsed ? (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Link
                    href="/pdf-chat"
                    className={cn(
                      "flex h-9 w-9 items-center justify-center rounded-lg transition-colors",
                      pathname === "/pdf-chat"
                        ? "bg-accent text-accent-foreground"
                        : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
                    )}
                    aria-label="Chat với PDF"
                  >
                    <FileTextIcon className="h-4 w-4" />
                  </Link>
                </TooltipTrigger>
                <TooltipContent side="right">Chat với PDF</TooltipContent>
              </Tooltip>
            ) : (
              <Link
                href="/pdf-chat"
                className={cn(
                  "flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                  pathname === "/pdf-chat"
                    ? "bg-accent text-accent-foreground"
                    : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
                )}
              >
                <FileTextIcon className="h-4 w-4 shrink-0" />
                Chat với PDF
              </Link>
            )}
          </div>

          {/* AI Sandbox link */}
          <div className={cn("px-2 pb-2", collapsed && "flex justify-center")}>
            {collapsed ? (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Link
                    href="/sandbox"
                    className={cn(
                      "flex h-9 w-9 items-center justify-center rounded-lg transition-colors",
                      pathname === "/sandbox"
                        ? "bg-accent text-accent-foreground"
                        : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
                    )}
                    aria-label="AI sandbox"
                  >
                    <FlaskConicalIcon className="h-4 w-4" />
                  </Link>
                </TooltipTrigger>
                <TooltipContent side="right">AI sandbox</TooltipContent>
              </Tooltip>
            ) : (
              <Link
                href="/sandbox"
                className={cn(
                  "flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                  pathname === "/sandbox"
                    ? "bg-accent text-accent-foreground"
                    : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
                )}
              >
                <FlaskConicalIcon className="h-4 w-4 shrink-0" />
                AI sandbox
              </Link>
            )}
          </div>

          <Separator />

          {/* Session history */}
          <ScrollArea className="flex-1 min-h-0">
            <div className="px-2 py-2">
              {!collapsed && (
                <div className="flex items-center justify-between px-2 pb-1.5">
                  {sessions.length > 0 ? (
                    <p className="text-[0.7rem] font-semibold uppercase tracking-widest text-muted-foreground">
                      Recent
                    </p>
                  ) : (
                    <p className="text-xs text-muted-foreground">
                      No sessions yet.
                    </p>
                  )}
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Link
                        href="/sessions"
                        className={cn(
                          "flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[0.68rem] font-medium transition-colors",
                          pathname === "/sessions"
                            ? "text-primary"
                            : "text-muted-foreground hover:text-foreground hover:bg-accent",
                        )}
                      >
                        <LayoutListIcon className="h-3 w-3" />
                        View all
                      </Link>
                    </TooltipTrigger>
                    <TooltipContent side="right">All sessions</TooltipContent>
                  </Tooltip>
                </div>
              )}
              {collapsed && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Link
                      href="/sessions"
                      className={cn(
                        "flex h-9 w-9 items-center justify-center rounded-lg transition-colors mx-auto mb-0.5",
                        pathname === "/sessions"
                          ? "bg-accent text-accent-foreground"
                          : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
                      )}
                      aria-label="All sessions"
                    >
                      <LayoutListIcon className="h-4 w-4" />
                    </Link>
                  </TooltipTrigger>
                  <TooltipContent side="right">All sessions</TooltipContent>
                </Tooltip>
              )}
              <div className="grid gap-0.5">
                {sessions.map((session) => {
                  const isActive = pathname === `/sessions/${session.id}`;
                  const isDeleting =
                    deleteMutation.isPending &&
                    deleteMutation.variables === session.id;
                  const topicDisplay = getSessionTopicDisplay(session.topic);
                  return collapsed ? (
                    <Tooltip key={session.id}>
                      <TooltipTrigger asChild>
                        <Link
                          href={`/sessions/${session.id}`}
                          className={cn(
                            "flex h-9 w-9 items-center justify-center rounded-lg transition-colors mx-auto",
                            isActive
                              ? "bg-accent text-accent-foreground"
                              : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
                          )}
                          aria-label={normalizeRawQuery(session.raw_query)}
                        >
                          <BookOpenIcon className="h-4 w-4" />
                        </Link>
                      </TooltipTrigger>
                      <TooltipContent side="right" className="max-w-56">
                        <p className="font-semibold text-xs line-clamp-3">
                          {normalizeRawQuery(session.raw_query)}
                        </p>
                        <p
                          className={cn(
                            "mt-0.5 min-h-[1rem] text-xs leading-snug",
                            topicDisplay.isPlaceholder
                              ? "italic text-muted-foreground/60"
                              : "text-muted-foreground",
                          )}
                        >
                          {topicDisplay.label}
                        </p>
                        <p className="text-[0.65rem] text-muted-foreground mt-1">
                          {formatCreatedAt(session.created_at)}
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  ) : (
                    <div
                      key={session.id}
                      className="group relative flex items-center"
                    >
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Link
                            href={`/sessions/${session.id}`}
                            className={cn(
                              "flex flex-1 min-w-0 flex-col rounded-lg px-2.5 py-2 text-left transition-colors pr-8",
                              isActive
                                ? "bg-accent text-accent-foreground"
                                : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
                            )}
                          >
                            <span className="text-xs font-semibold text-foreground truncate leading-snug">
                              {normalizeRawQuery(session.raw_query)}
                            </span>
                            <span
                              className={cn(
                                "mt-0.5 block min-h-[0.85rem] truncate text-[0.68rem] leading-snug",
                                topicDisplay.isPlaceholder
                                  ? "italic text-muted-foreground/60"
                                  : "text-muted-foreground",
                              )}
                            >
                              {topicDisplay.label}
                            </span>
                          </Link>
                        </TooltipTrigger>
                        <TooltipContent side="right" className="max-w-56">
                          <p className="text-xs text-muted-foreground">
                            {formatCreatedAt(session.created_at)}
                          </p>
                        </TooltipContent>
                      </Tooltip>

                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <button
                            type="button"
                            disabled={isDeleting}
                            className="absolute right-1.5 flex h-6 w-6 items-center justify-center rounded-md text-muted-foreground opacity-0 group-hover:opacity-100 hover:bg-destructive/10 hover:text-destructive transition-all disabled:opacity-50"
                            aria-label="Delete session"
                          >
                            <Trash2Icon className="h-3.5 w-3.5" />
                          </button>
                        </AlertDialogTrigger>
                        <AlertDialogContent size="sm">
                          <AlertDialogHeader>
                            <AlertDialogTitle>Delete session?</AlertDialogTitle>
                            <AlertDialogDescription>
                              <span className="font-medium text-foreground">
                                {normalizeRawQuery(session.raw_query)}
                              </span>{" "}
                              will be permanently removed. This cannot be
                              undone.
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel size="sm">
                              Cancel
                            </AlertDialogCancel>
                            <AlertDialogAction
                              size="sm"
                              variant="destructive"
                              onClick={() => deleteSession(session.id)}
                            >
                              Delete
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </div>
                  );
                })}
              </div>
            </div>
          </ScrollArea>

          <Separator />

          {/* User / logout */}
          <div
            className={cn(
              "flex items-center px-3 py-3",
              collapsed && "justify-center",
            )}
          >
            <Popover>
              <PopoverTrigger asChild>
                {collapsed ? (
                  <button
                    type="button"
                    className="flex h-8 w-8 items-center justify-center rounded-full hover:bg-muted transition-colors"
                    aria-label="Account menu"
                  >
                    <Avatar size="default">
                      <AvatarFallback>
                        {getInitials(user.full_name)}
                      </AvatarFallback>
                    </Avatar>
                  </button>
                ) : (
                  <button
                    type="button"
                    className="flex w-full items-center gap-2.5 rounded-lg px-1 py-1 hover:bg-muted transition-colors text-left"
                    aria-label="Account menu"
                  >
                    <Avatar size="default">
                      <AvatarFallback>
                        {getInitials(user.full_name)}
                      </AvatarFallback>
                    </Avatar>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-semibold text-foreground truncate">
                        {user.full_name}
                      </p>
                      <p className="text-[0.65rem] text-muted-foreground truncate">
                        {user.email}
                      </p>
                    </div>
                  </button>
                )}
              </PopoverTrigger>

              <PopoverContent
                side="right"
                align="end"
                sideOffset={8}
                className="w-56"
              >
                <PopoverHeader>
                  <PopoverTitle>{user.full_name}</PopoverTitle>
                  <PopoverDescription>{user.email}</PopoverDescription>
                </PopoverHeader>

                <Separator />

                <button
                  type="button"
                  className="flex w-full items-center gap-2.5 rounded-md px-2 py-2 text-sm text-destructive hover:bg-destructive/8 transition-colors"
                  onClick={() => {
                    clearAccessToken();
                    queryClient.removeQueries({ queryKey: ["accessToken"] });
                    queryClient.clear();
                    router.replace("/login");
                  }}
                >
                  <LogOutIcon className="h-4 w-4 shrink-0" />
                  Sign out
                </button>
              </PopoverContent>
            </Popover>
          </div>
        </aside>

        {/* Main content */}
        <div className="flex flex-1 flex-col min-w-0">
          <main className="flex-1 overflow-auto">{children}</main>
        </div>
      </div>
    </TooltipProvider>
  );
}
