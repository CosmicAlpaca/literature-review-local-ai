"use client";

import type {
  InputHTMLAttributes,
  ReactNode,
  TextareaHTMLAttributes,
} from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

type BaseProps = {
  label: string;
  hint?: string;
  error?: string;
};

type InputProps = BaseProps &
  InputHTMLAttributes<HTMLInputElement> & {
    as?: "input";
  };

type TextareaProps = BaseProps &
  TextareaHTMLAttributes<HTMLTextAreaElement> & {
    as: "textarea";
  };

type Props = InputProps | TextareaProps;

function FieldMessage({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <p className={cn("text-xs text-muted-foreground", className)}>{children}</p>
  );
}

export default function Field(props: Props) {
  const { label, hint, error } = props;

  if (props.as === "textarea") {
    const {
      as: _as,
      label: _label,
      hint: _hint,
      error: _error,
      className,
      id,
      name,
      ...controlProps
    } = props;
    const fieldId = id ?? name ?? label.toLowerCase().replace(/\s+/g, "-");

    return (
      <div className="grid gap-1.5">
        <Label htmlFor={fieldId}>{label}</Label>
        <Textarea
          id={fieldId}
          name={name}
          aria-invalid={Boolean(error)}
          className={cn("min-h-32 resize-y", className)}
          {...controlProps}
        />
        {error ? (
          <FieldMessage className="text-destructive">{error}</FieldMessage>
        ) : hint ? (
          <FieldMessage>{hint}</FieldMessage>
        ) : null}
      </div>
    );
  }

  const {
    label: _label,
    hint: _hint,
    error: _error,
    className,
    id,
    name,
    ...controlProps
  } = props;
  const fieldId = id ?? name ?? label.toLowerCase().replace(/\s+/g, "-");

  return (
    <div className="grid gap-1.5">
      <Label htmlFor={fieldId}>{label}</Label>
      <Input
        id={fieldId}
        name={name}
        aria-invalid={Boolean(error)}
        className={cn("h-10", className)}
        {...controlProps}
      />
      {error ? (
        <FieldMessage className="text-destructive">{error}</FieldMessage>
      ) : hint ? (
        <FieldMessage>{hint}</FieldMessage>
      ) : null}
    </div>
  );
}
