type Props = {
  title: string;
  body: string;
};

export default function EmptyState({ title, body }: Props) {
  return (
    <div className="py-4">
      <p className="text-sm font-semibold text-foreground mb-1">{title}</p>
      <p className="text-sm text-muted-foreground">{body}</p>
    </div>
  );
}
