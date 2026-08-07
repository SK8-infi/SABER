import { ReactNode } from "react";

import { siteConfig } from "@/config/site";
import { cn } from "@/lib/utils";

import LogoSvg from "@/assets/svg/logo";
import {
  Footer,
  FooterBottom,
  FooterColumn,
  FooterContent,
} from "../../ui/footer";
import { ModeToggle } from "../../ui/mode-toggle";

interface FooterLink {
  text: string;
  href: string;
}

interface FooterColumnProps {
  title: string;
  links: FooterLink[];
}

interface FooterProps {
  logo?: ReactNode;
  name?: string;
  columns?: FooterColumnProps[];
  copyright?: string;
  policies?: FooterLink[];
  showModeToggle?: boolean;
  className?: string;
}

export default function FooterSection({
  logo = <LogoSvg className="size-7" />,
  name = "SABER",
  columns = [
    {
      title: "Project",
      links: [
        { text: "GitHub Repository", href: siteConfig.links.github },
        { text: "Documentation", href: `${siteConfig.links.github}/tree/master/docs` },
        { text: "Training Guide", href: `${siteConfig.links.github}/blob/master/docs/TRAINING_DOCUMENTATION.md` },
      ],
    },
    {
      title: "Dashboard",
      links: [
        { text: "Live Query Inspector", href: "/dashboard/format/query" },
        { text: "Embedding Space", href: "/dashboard/format/embeddings" },
        { text: "Training Telemetry", href: "/dashboard/format/training" },
      ],
    },
    {
      title: "Team Sentinel8",
      links: [
        { text: "ISRO BAH 2026", href: siteConfig.links.github },
        { text: "Problem Statement 11", href: siteConfig.links.github },
        { text: "GitHub", href: siteConfig.links.github },
      ],
    },
  ],
  copyright = `© 2026 Team Sentinel8 · ISRO BAH 2026 · PS-11. All rights reserved.`,
  policies = [
    { text: "MIT License", href: `${siteConfig.links.github}/blob/master/LICENSE` },
  ],
  showModeToggle = true,
  className,
}: FooterProps) {
  return (
    <footer className={cn("bg-background w-full px-4", className)}>
      <div className="max-w-container mx-auto">
        <Footer>
          <FooterContent>
            <FooterColumn className="col-span-2 sm:col-span-3 md:col-span-1">
              <div className="flex items-center gap-2">
                {logo}
                <h3 className="text-xl font-bold">{name}</h3>
              </div>
            </FooterColumn>
            {columns.map((column) => (
              <FooterColumn key={column.title}>
                <h3 className="text-md pt-1 font-semibold">{column.title}</h3>
                {column.links.map((link) => (
                  <a
                    key={`${link.href}-${link.text}`}
                    href={link.href}
                    className="text-muted-foreground text-sm"
                  >
                    {link.text}
                  </a>
                ))}
              </FooterColumn>
            ))}
          </FooterContent>
          <FooterBottom>
            <div>{copyright}</div>
            <div className="flex items-center gap-4">
              {policies.map((policy) => (
                <a key={`${policy.href}-${policy.text}`} href={policy.href}>
                  {policy.text}
                </a>
              ))}
              {showModeToggle && <ModeToggle />}
            </div>
          </FooterBottom>
        </Footer>
      </div>
    </footer>
  );
}
