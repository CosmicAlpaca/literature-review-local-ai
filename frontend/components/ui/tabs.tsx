"use client";

import * as React from "react";
import { Tabs as TabsNamespace } from "radix-ui";

import { cn } from "@/lib/utils";

function Tabs({
  className,
  ...props
}: React.ComponentProps<typeof TabsNamespace.Root>) {
  return (
    <TabsNamespace.Root
      data-slot="tabs"
      className={cn("flex flex-col gap-3", className)}
      {...props}
    />
  );
}

function TabsList({
  className,
  ...props
}: React.ComponentProps<typeof TabsNamespace.List>) {
  return (
    <TabsNamespace.List
      data-slot="tabs-list"
      className={cn(
        "inline-flex h-9 w-fit max-w-full flex-wrap items-center justify-start gap-1 rounded-lg border border-border bg-secondary p-1 shadow-inner",
        className,
      )}
      {...props}
    />
  );
}

function TabsTrigger({
  className,
  ...props
}: React.ComponentProps<typeof TabsNamespace.Trigger>) {
  return (
    <TabsNamespace.Trigger
      data-slot="tabs-trigger"
      className={cn(
        "inline-flex items-center justify-center whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
        "text-foreground/65 hover:bg-background/70 hover:text-foreground",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
        "disabled:pointer-events-none disabled:opacity-50",
        "data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:shadow-md data-[state=active]:ring-1 data-[state=active]:ring-border/80 data-[state=active]:font-semibold",
        className,
      )}
      {...props}
    />
  );
}

function TabsContent({
  className,
  ...props
}: React.ComponentProps<typeof TabsNamespace.Content>) {
  return (
    <TabsNamespace.Content
      data-slot="tabs-content"
      className={cn(
        "mt-1 flex-1 outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
        className,
      )}
      {...props}
    />
  );
}

export { Tabs, TabsList, TabsTrigger, TabsContent };
