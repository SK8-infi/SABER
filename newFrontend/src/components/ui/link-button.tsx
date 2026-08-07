import { type VariantProps } from "class-variance-authority";
import { type ComponentProps, type ReactNode } from "react";

import { buttonVariants } from "./button";
import { cn } from "@/lib/utils";

export interface LinkButtonProps {
  href: string;
  children: ReactNode;
  variant?: VariantProps<typeof buttonVariants>["variant"];
  icon?: ReactNode;
  iconRight?: ReactNode;
  size?: ComponentProps<typeof buttonVariants>["size"];
  className?: string;
}

export function LinkButton({
  href,
  children,
  variant = "default",
  icon,
  iconRight,
  size = "lg",
  className,
}: LinkButtonProps) {
  return (
    <a
      href={href}
      className={cn(buttonVariants({ variant, size, className }))}
    >
      {icon}
      {children}
      {iconRight}
    </a>
  );
}
