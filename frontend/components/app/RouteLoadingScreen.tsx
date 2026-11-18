import { Spinner } from "@/components/ui/spinner";

export default function RouteLoadingScreen() {
  return (
    <div
      className="flex min-h-screen items-center justify-center bg-background"
      aria-busy="true"
      aria-live="polite"
    >
      <Spinner className="size-8 text-muted-foreground" />
    </div>
  );
}
