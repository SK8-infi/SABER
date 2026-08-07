"use client";

import { ChevronsUpDownIcon } from "lucide-react";
import { useTheme } from "next-themes";

import { buttonVariants } from "./button";
import { cn } from "@/lib/utils";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "./dropdown-menu";

export function ModeToggle() {
  const { theme, setTheme } = useTheme();
  const selectedTheme = theme ?? "dark";

  return (
    <DropdownMenu>
      <DropdownMenuTrigger className={cn(buttonVariants({ variant: "ghost" }), "gap-1 px-2 py-0 text-xs cursor-pointer")}>
        <span className="capitalize" suppressHydrationWarning>
          {selectedTheme}
        </span>
        <span className="inline"> theme</span>
        <ChevronsUpDownIcon className="size-3" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={() => setTheme("light")}>
          Light
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => setTheme("dark")}>
          Dark
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => setTheme("system")}>
          System
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
