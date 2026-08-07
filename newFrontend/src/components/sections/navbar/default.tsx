"use client";

import { type VariantProps } from "class-variance-authority";
import { Menu } from "lucide-react";
import { ReactNode } from "react";

import { siteConfig } from "@/config/site";
import { cn } from "@/lib/utils";

import LogoSvg from "@/assets/svg/logo";
import ModeToggle from "@/components/layout/ModeToggle";
import { Button, buttonVariants } from "../../ui/button";
import {
  Navbar as NavbarComponent,
  NavbarLeft,
  NavbarRight,
} from "../../ui/navbar";
import { Sheet, SheetContent, SheetTitle, SheetTrigger } from "../../ui/sheet";

interface NavbarLink {
  text: string;
  href: string;
}

interface NavbarActionProps {
  text: string;
  href: string;
  variant?: VariantProps<typeof buttonVariants>["variant"];
  icon?: ReactNode;
  iconRight?: ReactNode;
  isButton?: boolean;
}

interface NavbarProps {
  logo?: ReactNode;
  name?: string;
  homeUrl?: string;
  mobileLinks?: NavbarLink[];
  actions?: NavbarActionProps[];
  showNavigation?: boolean;
  customNavigation?: ReactNode;
  className?: string;
}

export default function Navbar({
  logo = <LogoSvg className="size-8" />,
  name = "SABER",
  homeUrl = "/",
  mobileLinks = [
    { text: "Interactive Query Space", href: "/dashboard/format/embeddings" },
    { text: "Classic Query Inspector", href: "/dashboard/format/query" },
    { text: "Ablation Studies", href: "/dashboard/format/abliation" },
    { text: "Training Telemetry", href: "/dashboard/format/training" },
  ],
  actions = [
    {
      text: "Launch Dashboard",
      href: "/dashboard/format/embeddings",
      isButton: true,
      variant: "default",
    },
  ],
  showNavigation = true,
  customNavigation,
  className,
}: NavbarProps) {
  const dashboardLinks = [
    { text: "Query Space", href: "/dashboard/format/embeddings" },
    { text: "Query Inspector", href: "/dashboard/format/query" },
    { text: "Ablation", href: "/dashboard/format/abliation" },
    { text: "Training", href: "/dashboard/format/training" },
  ];

  return (
    <header className={cn("sticky top-0 z-50 -mb-4 px-4 pb-4", className)}>
      <div className="fade-bottom bg-background/15 absolute left-0 h-24 w-full backdrop-blur-lg"></div>
      <div className="max-w-container relative mx-auto">
        <NavbarComponent>
          <NavbarLeft>
            <a href={homeUrl} className="flex items-center gap-2 text-xl font-bold shrink-0">
              {logo}
              {name}
            </a>
            {/* Dashboard format links right after logo */}
            <nav className="hidden items-center gap-1 md:flex ml-2">
              {dashboardLinks.map((link) => (
                <a
                  key={link.href}
                  href={link.href}
                  className="rounded-md px-3 py-1.5 text-sm text-muted-foreground hover:text-foreground hover:bg-muted/60 transition-colors"
                >
                  {link.text}
                </a>
              ))}
            </nav>
          </NavbarLeft>
          <NavbarRight>
            <ModeToggle />
            {actions.map((action) =>
              action.isButton ? (
                <a
                  key={`${action.href}-${action.text}`}
                  href={action.href}
                  className={cn(buttonVariants({ variant: action.variant || "default" }))}
                >
                  {action.icon}
                  {action.text}
                  {action.iconRight}
                </a>
              ) : (
                <a
                  key={`${action.href}-${action.text}`}
                  href={action.href}
                  className="hidden text-sm md:block"
                >
                  {action.text}
                </a>
              ),
            )}
            <Sheet>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon" className="shrink-0 md:hidden">
                  <Menu className="size-5" />
                  <span className="sr-only">Toggle navigation menu</span>
                </Button>
              </SheetTrigger>
              <SheetContent side="right">
                <SheetTitle className="sr-only">Navigation menu</SheetTitle>
                <nav className="grid gap-6 text-lg font-medium">
                  <a href={homeUrl} className="flex items-center gap-2 text-xl font-bold">
                    <span>{name}</span>
                  </a>
                  {mobileLinks.map((link) => (
                    <a
                      key={`${link.href}-${link.text}`}
                      href={link.href}
                      className="text-muted-foreground hover:text-foreground"
                    >
                      {link.text}
                    </a>
                  ))}
                </nav>
              </SheetContent>
            </Sheet>
          </NavbarRight>
        </NavbarComponent>
      </div>
    </header>
  );
}
