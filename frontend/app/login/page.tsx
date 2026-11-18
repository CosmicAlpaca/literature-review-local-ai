"use client";

import { ReactNode, useState } from "react";
import { useRouter } from "next/navigation";
import { useForm } from "react-hook-form";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import toast from "react-hot-toast";
import { z } from "zod";
import { Button } from "@/components/ui/Button";
import Field from "@/components/ui/Field";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import AppBrand from "@/components/app/AppBrand";
import { setAccessToken } from "@/lib/auth";
import { apiFetch } from "@/lib/axios";

const loginSchema = z.object({
  email: z.email("Enter a valid email address."),
  password: z.string().min(1, "Password is required."),
});

const registerSchema = loginSchema.extend({
  fullName: z.string().min(2, "Full name must be at least 2 characters."),
});

type FormValues = {
  fullName: string;
  email: string;
  password: string;
};

export default function LoginPage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [mode, setMode] = useState<"login" | "register">("login");
  const {
    register,
    handleSubmit,
    formState: { errors },
    setError,
    clearErrors,
    reset,
  } = useForm<FormValues>({
    defaultValues: { fullName: "", email: "", password: "" },
  });

  const loginMutation = useMutation({
    mutationFn: (values: FormValues) =>
      apiFetch("/auth/login", {
        method: "POST",
        body: JSON.stringify({
          email: values.email,
          password: values.password,
        }),
      }),
    onSuccess: (data) => {
      const authData = data as { access_token: string };
      setAccessToken(authData.access_token);
      queryClient.setQueryData(["accessToken"], authData.access_token);
      toast.success("Signed in successfully.");
      router.push("/new-search");
    },
    onError: (err) => {
      toast.error(err instanceof Error ? err.message : "Unable to sign in.");
    },
  });

  const registerMutation = useMutation({
    mutationFn: (values: FormValues) =>
      apiFetch("/auth/register", {
        method: "POST",
        body: JSON.stringify({
          full_name: values.fullName,
          email: values.email,
          password: values.password,
        }),
      }),
    onSuccess: (data) => {
      const authData = data as { access_token: string };

      setAccessToken(authData.access_token);
      queryClient.setQueryData(["accessToken"], authData.access_token);
      toast.success("Account created successfully.");
      router.push("/new-search");
    },
    onError: (err) => {
      toast.error(
        err instanceof Error ? err.message : "Unable to create account.",
      );
    },
  });

  const activeMutation = mode === "login" ? loginMutation : registerMutation;
  const isPending = activeMutation.isPending;

  function onSubmit(values: FormValues) {
    const parseResult =
      mode === "register"
        ? registerSchema.safeParse(values)
        : loginSchema.safeParse(values);

    if (!parseResult.success) {
      clearErrors();
      for (const issue of parseResult.error.issues) {
        const field = issue.path[0];
        if (field === "fullName" || field === "email" || field === "password") {
          setError(field, { type: "zod", message: issue.message });
        }
      }
      return;
    }

    activeMutation.mutate(values);
  }

  return (
    <Layout
      left={
        <div className="relative z-10 flex flex-col justify-between h-full">
          <div className="max-w-[34rem]">
            <AppBrand tone="inverse" href="/" className="mb-8" />
            <h1 className="font-serif text-5xl font-normal leading-[0.95] tracking-tight xl:text-6xl">
              Your literature review workspace
            </h1>

            <p className="mt-5 text-base leading-relaxed text-white/80">
              Sign in to continue your research workflow, review candidate
              papers, and manage your saved sessions in one place.
            </p>

            <ul className="mt-8 grid gap-3.5">
              {[
                "Run and manage review sessions.",
                "Track candidate selection history.",
                "Keep your workspace organized across projects.",
              ].map((item) => (
                <li key={item} className="flex items-start gap-3 text-white/80">
                  <span className="mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full bg-[#d4b56f] shadow-[0_0_0_5px_rgba(212,181,111,0.16)]" />
                  <span className="text-sm leading-relaxed">{item}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      }
      right={
        <Card className="w-full max-w-[400px]">
          <CardHeader>
            <div className="inline-flex w-fit items-center rounded-full border border-primary/15 bg-primary/8 px-3 py-1 text-xs font-semibold uppercase tracking-widest text-primary">
              {mode === "login" ? "Sign in" : "Create researcher account"}
            </div>
            <CardTitle className="text-2xl font-serif font-normal tracking-tight mt-2">
              {mode === "login"
                ? "Access your review workspace"
                : "Set up your review workspace"}
            </CardTitle>
            <CardDescription>
              {mode === "login"
                ? "Use your existing account to continue reviewing sessions, candidates, and trusted results."
                : "Create a user account to start new literature review sessions against the current backend."}
            </CardDescription>
          </CardHeader>

          <CardContent>
            <form onSubmit={handleSubmit(onSubmit)} className="grid gap-4">
              {mode === "register" && (
                <Field
                  label="Full name"
                  autoComplete="name"
                  placeholder="Your full name"
                  error={errors.fullName?.message}
                  {...register("fullName")}
                />
              )}

              <Field
                label="Email"
                type="email"
                autoComplete="email"
                placeholder="Enter your email"
                error={errors.email?.message}
                {...register("email")}
              />

              <Field
                label="Password"
                type="password"
                autoComplete={
                  mode === "login" ? "current-password" : "new-password"
                }
                placeholder={
                  mode === "login"
                    ? "Enter your password"
                    : "Create a secure password"
                }
                error={errors.password?.message}
                {...register("password")}
              />

              <Button
                type="submit"
                className="w-full mt-1"
                disabled={isPending}
              >
                {isPending
                  ? mode === "login"
                    ? "Signing in…"
                    : "Creating account…"
                  : mode === "login"
                    ? "Sign in"
                    : "Create account"}
              </Button>
            </form>

            <Separator className="my-5" />

            <p className="text-center text-sm text-muted-foreground">
              {mode === "login"
                ? "Need an account?"
                : "Already have an account?"}{" "}
              <button
                type="button"
                className="font-semibold text-primary hover:underline"
                onClick={() => {
                  clearErrors();
                  reset();
                  setMode(mode === "login" ? "register" : "login");
                }}
              >
                {mode === "login" ? "Register here" : "Sign in here"}
              </button>
            </p>
          </CardContent>
        </Card>
      }
    />
  );
}

type LayoutProps = {
  left: ReactNode;
  right: ReactNode;
};

function Layout({ left, right }: LayoutProps) {
  return (
    <div className="grid min-h-screen lg:grid-cols-[1.1fr_0.9fr]">
      {/* Brand panel */}
      <div className="relative hidden flex-col justify-between gap-10 overflow-hidden bg-[#102133] p-14 text-white lg:flex">
        {/* Decorative gradient blobs */}
        <div className="pointer-events-none absolute inset-0">
          <div className="absolute left-0 top-0 h-72 w-72 -translate-x-1/2 -translate-y-1/2 rounded-full bg-[#71afc2]/20 blur-3xl" />
          <div className="absolute bottom-0 right-0 h-56 w-56 translate-x-1/4 translate-y-1/4 rounded-full bg-[#faebc7]/10 blur-3xl" />
        </div>
        {left}
      </div>

      {/* Form panel */}
      <div className="flex flex-col items-center justify-center bg-gradient-to-b from-[rgba(255,253,248,0.92)] to-white px-6 py-12">
        <AppBrand href="/" className="mb-8 lg:hidden" />
        {right}
      </div>
    </div>
  );
}
