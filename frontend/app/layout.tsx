import type { Metadata } from "next";
import type { ReactNode } from "react";
import { Toaster } from "react-hot-toast";
import "./globals.css";
import { Roboto } from "next/font/google";
import { cn } from "@/lib/utils";
import Providers from "@/components/providers";
import AuthShell from "@/components/app/AuthShell";

const roboto = Roboto({ subsets: ["latin"], variable: "--font-sans" });

export const metadata: Metadata = {
  title: "Literature Review System",
  description:
    "Search academic papers, shortlist evidence, and draft literature reviews with citation checks.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: ReactNode;
}>) {
  return (
    <html lang="en" className={cn("font-sans", roboto.variable)}>
      <body className="antialiased">
        <Providers>
          <AuthShell>{children}</AuthShell>
        </Providers>
        <Toaster position="top-right" />
      </body>
    </html>
  );
}
