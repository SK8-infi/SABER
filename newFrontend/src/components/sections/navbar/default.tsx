"use client";

import { Menu, Satellite } from "lucide-react";
import { Button } from "../../ui/button";
import {
  Navbar as NavbarComponent,
  NavbarLeft,
  NavbarRight,
} from "../../ui/navbar";
import { Sheet, SheetContent, SheetTitle, SheetTrigger } from "../../ui/sheet";

const NAV_LINKS = [
  { text: "Use Cases", href: "#use-cases" },
  { text: "Architecture", href: "#architecture" },
  { text: "Benchmarks", href: "#benchmarks" },
  { text: "FAQ", href: "#faq" },
];

export default function Navbar() {
  return (
    <header className="sticky top-0 z-50 -mb-4 px-4 pb-4">
      <div className="fade-bottom bg-background/15 absolute left-0 h-24 w-full backdrop-blur-lg" />
      <div className="max-w-container relative mx-auto">
        <NavbarComponent>
          <NavbarLeft>
            <a href="/" className="flex items-center gap-2.5 text-xl font-bold">
              <div className="flex items-center justify-center size-8 rounded-lg bg-[#FBBA72]/15 border border-[#FBBA72]/30">
                <Satellite className="size-4 text-[#FBBA72]" />
              </div>
              <span>SABER</span>
              <span className="hidden text-xs font-medium text-muted-foreground sm:block">
                ISRO BAH 2026
              </span>
            </a>
            <nav className="hidden items-center gap-6 md:flex ml-6">
              {NAV_LINKS.map(l => (
                <a key={l.text} href={l.href}
                  className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                  {l.text}
                </a>
              ))}
            </nav>
          </NavbarLeft>
          <NavbarRight>
            <a href="https://github.com/SK8-infi/SABER"
              target="_blank" rel="noopener noreferrer"
              className="hidden text-sm text-muted-foreground hover:text-foreground transition-colors md:block">
              GitHub
            </a>
            <Button variant="default" asChild>
              <a href="/dashboard/format/query">Launch Demo</a>
            </Button>
            {/* Mobile */}
            <Sheet>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon" className="shrink-0 md:hidden">
                  <Menu className="size-5" />
                  <span className="sr-only">Toggle menu</span>
                </Button>
              </SheetTrigger>
              <SheetContent side="right">
                <SheetTitle className="sr-only">Navigation</SheetTitle>
                <nav className="grid gap-6 text-lg font-medium pt-8">
                  <a href="/" className="flex items-center gap-2 font-bold">
                    <Satellite className="size-5 text-[#FBBA72]" /> SABER
                  </a>
                  {NAV_LINKS.map(l => (
                    <a key={l.text} href={l.href}
                      className="text-muted-foreground hover:text-foreground">{l.text}</a>
                  ))}
                  <a href="/dashboard/format/query"
                    className="text-[#FBBA72] font-semibold">Launch Demo →</a>
                </nav>
              </SheetContent>
            </Sheet>
          </NavbarRight>
        </NavbarComponent>
      </div>
    </header>
  );
}
