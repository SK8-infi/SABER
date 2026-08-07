"use client";

import Link from "next/link";
import * as React from "react";
import { ReactNode } from "react";

import { siteConfig } from "@/config/site";
import { cn } from "@/lib/utils";

import LogoSvg from "@/assets/svg/logo";
import LaunchUI from "../logos/launch-ui";
import {
  NavigationMenu,
  NavigationMenuContent,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  NavigationMenuTrigger,
  navigationMenuTriggerStyle,
} from "./navigation-menu";

interface ComponentItem {
  title: string;
  href: string;
  description: string;
}

interface MenuItem {
  title: string;
  href?: string;
  isLink?: boolean;
  content?: ReactNode;
}

interface NavigationProps {
  menuItems?: MenuItem[];
  components?: ComponentItem[];
  formatModules?: ComponentItem[];
  logo?: ReactNode;
  logoTitle?: string;
  logoDescription?: string;
  logoHref?: string;
  introItems?: {
    title: string;
    href: string;
    description: string;
  }[];
}

export default function Navigation({
  menuItems = [
    {
      title: "Results & Metrics",
      isLink: true,
      href: "/dashboard/format/training",
    },
    {
      title: "Dashboard Format",
      content: "format-modules",
    },
  ],
  formatModules = [
    {
      title: "Interactive Query Space",
      href: "/dashboard/format/embeddings",
      description: "2D metric-preserving manifold projection with real-time cross-modal retrieval.",
    },
    {
      title: "Classic Query Inspector",
      href: "/dashboard/format/query",
      description: "Single scene query engine inspecting candidate ranks & land-cover overlap.",
    },
    {
      title: "Ablation Studies",
      href: "/dashboard/format/abliation",
      description: "Component-wise ablation benchmarks comparing CFM ODE, LoRA, and DOFA.",
    },
    {
      title: "Training Telemetry",
      href: "/dashboard/format/training",
      description: "Real-time loss convergence curves, learning rates, and validation metrics.",
    },
  ],
  components = [
    {
      title: "01. Multi-Sensor Ingestion",
      href: "#architecture",
      description: "Ingests Sentinel-1 SAR & Sentinel-2 12-band optical tiles into wavelength-calibrated patches.",
    },
    {
      title: "02. Wavelength ViT & LoRA",
      href: "#architecture",
      description: "Dynamic DOFA patch hypernetworks with parameter-efficient LoRA adapters (1.82% trainable).",
    },
    {
      title: "03. CFM Latent ODE Bridge",
      href: "#architecture",
      description: "Neural vector-field ODE solving straight-line probability paths connecting SAR & Optical embeddings.",
    },
    {
      title: "04. FAISS Vector Search",
      href: "#architecture",
      description: "Sub-30ms similarity retrieval using IVF-PQ index with k-reciprocal graph re-ranking.",
    },
  ],
  logo = <LogoSvg className="size-10" />,
  logoTitle = "SABER Pipeline Flow",
  logoDescription = "Cross-modal satellite image retrieval unifying SAR radar & Optical imagery onto a metric-optimized hypersphere.",
  logoHref = "#architecture",
  introItems = [
    {
      title: "End-to-End Pipeline",
      href: "#architecture",
      description: "Explore the 4-stage pipeline translating SAR radar backscatter to optical spectral space.",
    },
    {
      title: "CFM Latent ODE Bridge",
      href: "#architecture",
      description: "Continuous Flow Matching probability ODE translating SAR to Optical latent space in 11.6ms.",
    },
    {
      title: "Sub-30ms Search",
      href: "/dashboard/format/embeddings",
      description: "Real-time FAISS ANN search with k-reciprocal land-cover graph re-ranking.",
    },
  ],
}: NavigationProps) {
  return (
    <NavigationMenu className="hidden md:flex">
      <NavigationMenuList>
        {menuItems.map((item) => (
          <NavigationMenuItem key={item.title}>
            {item.isLink ? (
              <NavigationMenuLink
                className={navigationMenuTriggerStyle()}
                asChild
              >
                <Link href={item.href || ""}>{item.title}</Link>
              </NavigationMenuLink>
            ) : (
              <>
                <NavigationMenuTrigger>{item.title}</NavigationMenuTrigger>
                <NavigationMenuContent>
                  {item.content === "default" ? (
                    <ul className="grid gap-3 p-4 md:w-[400px] lg:w-[500px] lg:grid-cols-[.75fr_1fr]">
                      <li className="row-span-3">
                        <NavigationMenuLink asChild>
                          <a
                            className="from-muted/30 to-muted/10 flex h-full w-full flex-col justify-end rounded-md bg-linear-to-b p-6 no-underline outline-hidden select-none focus:shadow-md"
                            href={logoHref}
                          >
                            {logo}
                            <div className="mt-4 mb-2 text-lg font-medium">
                              {logoTitle}
                            </div>
                            <p className="text-muted-foreground text-sm leading-tight">
                              {logoDescription}
                            </p>
                          </a>
                        </NavigationMenuLink>
                      </li>
                      {introItems.map((intro) => (
                        <ListItem
                          key={intro.title}
                          href={intro.href}
                          title={intro.title}
                        >
                          {intro.description}
                        </ListItem>
                      ))}
                    </ul>
                  ) : item.content === "format-modules" ? (
                    <ul className="grid w-[400px] gap-3 p-4 md:w-[500px] md:grid-cols-2 lg:w-[600px]">
                      {formatModules.map((module) => (
                        <ListItem
                          key={module.title}
                          title={module.title}
                          href={module.href}
                        >
                          {module.description}
                        </ListItem>
                      ))}
                    </ul>
                  ) : item.content === "components" ? (
                    <ul className="grid w-[400px] gap-3 p-4 md:w-[500px] md:grid-cols-2 lg:w-[600px]">
                      {components.map((component) => (
                        <ListItem
                          key={component.title}
                          title={component.title}
                          href={component.href}
                        >
                          {component.description}
                        </ListItem>
                      ))}
                    </ul>
                  ) : (
                    item.content
                  )}
                </NavigationMenuContent>
              </>
            )}
          </NavigationMenuItem>
        ))}
      </NavigationMenuList>
    </NavigationMenu>
  );
}

function ListItem({
  className,
  title,
  children,
  ...props
}: React.ComponentProps<"a"> & { title: string }) {
  return (
    <li>
      <NavigationMenuLink asChild>
        <a
          data-slot="list-item"
          className="hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground block space-y-1 rounded-md p-3 leading-none no-underline outline-hidden transition-colors select-none"
          {...props}
        >
          <div className="text-sm leading-none font-medium">{title}</div>
          <p className="text-muted-foreground line-clamp-2 text-sm leading-snug">
            {children}
          </p>
        </a>
      </NavigationMenuLink>
    </li>
  );
}
