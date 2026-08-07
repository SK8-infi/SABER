import { ArrowRightIcon } from "lucide-react";
import { ReactNode } from "react";

import { siteConfig } from "@/config/site";
import { cn } from "@/lib/utils";

import Github from "../../logos/github";
import { Badge } from "../../ui/badge";
import Glow from "../../ui/glow";
import { LinkButton, type LinkButtonProps } from "../../ui/link-button";
import { Mockup, MockupFrame } from "../../ui/mockup";
import Screenshot from "../../ui/screenshot";
import { Section } from "../../ui/section";

interface HeroButtonProps extends Omit<LinkButtonProps, "children"> {
  text: string;
}

interface HeroProps {
  title?: string;
  description?: string;
  mockup?: ReactNode | false;
  badge?: ReactNode | false;
  buttons?: HeroButtonProps[] | false;
  className?: string;
}

const DEFAULT_HERO_BUTTONS: HeroButtonProps[] = [
  {
    href: siteConfig.getStartedUrl,
    text: "Get Started",
    variant: "default",
  },
  {
    href: "https://github.com/SK8-infi/SABER",
    text: "GitHub",
    variant: "glow",
    icon: <Github className="mr-2 size-4" />,
  },
];

const DEFAULT_HERO_BADGE = (
  <div className="animate-appear flex flex-wrap items-center justify-center gap-2">
    <Badge variant="outline">ISRO BAH 2026</Badge>
    <Badge variant="outline">PS-11</Badge>
    <Badge variant="outline">Team Sentinel8</Badge>
  </div>
);

const DEFAULT_HERO_MOCKUP = (
  <Screenshot
    srcLight="/dashboard-light.png"
    srcDark="/dashboard-dark.png"
    alt="Launch UI app screenshot"
    width={1248}
    height={765}
    loading="eager"
    className="w-full"
  />
);

export default function Hero({
  title = "SABER — Sensor-Agnostic Bridged Embedding Retrieval",
  description = "Cross-modal satellite image retrieval. SAR & Multispectral EO modalities unified onto a metric-optimised hypersphere via wavelength hypernetworks, LoRA adapters, and Conditional Flow Matching ODE latent bridges.",
  mockup = DEFAULT_HERO_MOCKUP,
  badge = DEFAULT_HERO_BADGE,
  buttons = DEFAULT_HERO_BUTTONS,
  className,
}: HeroProps) {
  return (
    <Section
      className={cn(
        "fade-bottom overflow-hidden pt-10 sm:pt-14 pb-0 sm:pb-0 md:pb-0",
        className,
      )}
    >
      <div className="max-w-container mx-auto flex flex-col gap-8 pt-4 sm:gap-12">
        <div className="flex flex-col items-center gap-4 text-center sm:gap-6">
          {badge !== false && badge}
          <h1 className="animate-appear from-foreground to-foreground dark:to-muted-foreground relative z-10 inline-block bg-linear-to-r bg-clip-text text-3xl font-semibold tracking-tight text-balance text-transparent drop-shadow-2xl sm:text-5xl md:text-6xl leading-tight sm:leading-tight md:leading-[1.15]">
            {title}
          </h1>
          <p className="text-sm animate-appear text-muted-foreground relative z-10 max-w-[850px] font-medium text-balance opacity-0 delay-100 sm:text-lg">
            {description}
          </p>
          {buttons !== false && buttons.length > 0 && (
            <div className="animate-appear relative z-10 flex justify-center gap-4 opacity-0 delay-300">
              {buttons.map((button) => (
                <LinkButton
                  key={`${button.href}-${button.text}`}
                  variant={button.variant || "default"}
                  size="lg"
                  href={button.href}
                  icon={button.icon}
                  iconRight={button.iconRight}
                >
                  {button.text}
                </LinkButton>
              ))}
            </div>
          )}
          {mockup !== false && (
            <div className="relative w-full pt-12">
              <MockupFrame
                className="animate-appear opacity-0 delay-700"
                size="small"
              >
                <Mockup
                  type="responsive"
                  className="bg-background/90 w-full rounded-xl border-0"
                >
                  {mockup}
                </Mockup>
              </MockupFrame>
              <Glow
                variant="top"
                className="animate-appear-zoom opacity-100 delay-500"
              />
            </div>
          )}
        </div>
      </div>
    </Section>
  );
}
