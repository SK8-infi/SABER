import { cva, VariantProps } from "class-variance-authority";
import React from "react";

import { cn } from "@/lib/utils";

const glowVariants = cva("absolute w-full pointer-events-none z-0", {
  variants: {
    variant: {
      top: "top-0",
      above: "-top-[128px]",
      bottom: "bottom-0",
      below: "-bottom-[128px]",
      center: "top-[50%]",
    },
  },
  defaultVariants: {
    variant: "top",
  },
});

function Glow({
  className,
  variant,
  ...props
}: React.ComponentProps<"div"> & VariantProps<typeof glowVariants>) {
  return (
    <div
      data-slot="glow"
      className={cn(glowVariants({ variant }), className)}
      {...props}
    >
      <div
        className={cn(
          "from-brand/80 via-brand-foreground/60 to-transparent absolute left-1/2 h-[350px] w-[85%] -translate-x-1/2 scale-150 rounded-[50%] bg-radial from-0% to-75% opacity-85 blur-3xl sm:h-[600px] dark:opacity-100",
          variant === "center" && "-translate-y-1/2",
        )}
      />
      <div
        className={cn(
          "from-brand to-transparent absolute left-1/2 h-[220px] w-[65%] -translate-x-1/2 scale-140 rounded-[50%] bg-radial from-0% to-60% opacity-95 blur-2xl sm:h-[360px] dark:opacity-100",
          variant === "center" && "-translate-y-1/2",
        )}
      />
    </div>
  );
}

export default Glow;
