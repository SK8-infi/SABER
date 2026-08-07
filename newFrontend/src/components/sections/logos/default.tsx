import { ReactNode } from "react";

import { siteConfig } from "@/config/site";

import Cfm from "../../logos/cfm";
import Dofa from "../../logos/dofa";
import Faiss from "../../logos/faiss";
import FastAPI from "../../logos/fastapi";
import Lora from "../../logos/lora";
import Python from "../../logos/python";
import PyTorch from "../../logos/pytorch";
import React from "../../logos/react";
import ShadcnUi from "../../logos/shadcn-ui";
import Tailwind from "../../logos/tailwind";
import TypeScript from "../../logos/typescript";
import { Badge } from "../../ui/badge";
import Logo from "../../ui/logo";
import { Section } from "../../ui/section";

interface LogosProps {
  title?: string;
  badge?: ReactNode | false;
  logos?: ReactNode[] | false;
  className?: string;
}

export default function Logos({
  title = "Built with state-of-the-art AI models, frameworks, vector engines and modern web tools",
  badge = (
    <Badge variant="outline" className="border-brand/30 text-brand">
      SABER Tech Stack & AI Models
    </Badge>
  ),
  logos = [
    <Logo
      key="dofa"
      image={Dofa}
      name="DOFA ViT"
      version="Wavelength"
      badge="Foundation Model"
    />,
    <Logo
      key="cfm"
      image={Cfm}
      name="CFM Latent ODE"
      version="Vector Field"
      badge="ODE Bridge"
    />,
    <Logo
      key="lora"
      image={Lora}
      name="LoRA Adapters"
      version="PEFT"
      badge="1.82% Trainable"
    />,
    <Logo
      key="pytorch"
      image={PyTorch}
      name="PyTorch"
      version="2.5.1"
      badge="Core ML"
    />,
    <Logo
      key="python"
      image={Python}
      name="Python"
      version="3.11"
      badge="Backend"
    />,
    <Logo
      key="fastapi"
      image={FastAPI}
      name="FastAPI"
      version="0.115"
      badge="Async API"
    />,
    <Logo
      key="faiss"
      image={Faiss}
      name="FAISS"
      version="1.8.0"
      badge="Vector Index"
    />,
    <Logo
      key="react"
      image={React}
      name="React"
      version="19.2"
      badge="Frontend"
    />,
    <Logo
      key="typescript"
      image={TypeScript}
      name="TypeScript"
      version="5.9"
    />,
    <Logo
      key="shadcn"
      image={ShadcnUi}
      name="Shadcn/ui"
      version="4.13"
    />,
    <Logo
      key="tailwind"
      image={Tailwind}
      name="Tailwind CSS"
      version="4.0"
    />,
  ],
  className,
}: LogosProps) {
  return (
    <Section className={className}>
      <div className="max-w-container mx-auto flex flex-col items-center gap-8 text-center">
        <div className="flex flex-col items-center gap-6">
          {badge !== false && badge}
          <h2 className="text-md font-semibold sm:text-2xl">{title}</h2>
        </div>
        {logos !== false && logos.length > 0 && (
          <div className="flex flex-wrap items-center justify-center gap-8 sm:gap-10">
            {logos}
          </div>
        )}
      </div>
    </Section>
  );
}
