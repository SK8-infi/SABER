import { cva, VariantProps } from "class-variance-authority";
import React from "react";

import { cn } from "@/lib/utils";

const glowVariants = cva("absolute w-full", {
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
      className={cn(glowVariants({ variant }), "pointer-events-none z-0", className)}
      {...props}
    >
      <div
        className={cn(
          "from-brand/60 via-brand-foreground/40 to-transparent absolute left-1/2 h-[256px] w-[70%] -translate-x-1/2 scale-[2.5] rounded-[50%] bg-radial from-10% to-70% opacity-70 blur-xl sm:h-[512px] dark:opacity-90",
          variant === "center" && "-translate-y-1/2",
        )}
      />
      <div
        className={cn(
          "from-brand/80 to-transparent absolute left-1/2 h-[128px] w-[50%] -translate-x-1/2 scale-200 rounded-[50%] bg-radial from-10% to-60% opacity-80 blur-lg sm:h-[256px] dark:opacity-100",
          variant === "center" && "-translate-y-1/2",
        )}
      />
    </div>
  );
}

export default Glow;
