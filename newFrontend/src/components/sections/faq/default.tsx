import Link from "next/link";
import { ReactNode } from "react";

import { siteConfig } from "@/config/site";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "../../ui/accordion";
import { Section } from "../../ui/section";

interface FAQItemProps {
  question: string;
  answer: ReactNode;
  value?: string;
}

interface FAQProps {
  title?: string;
  items?: FAQItemProps[] | false;
  className?: string;
}

export default function FAQ({
  title = "Frequently Asked Questions",
  items = [
    {
      question: "What is SABER and what problem does it solve?",
      answer: (
        <>
          <p className="text-muted-foreground mb-4 max-w-[640px] text-balance">
            SABER (Sensor-Agnostic Bridged Embedding Retrieval) is a
            cross-modal satellite image retrieval framework developed for ISRO
            BAH 2026 (Problem Statement 11).
          </p>
          <p className="text-muted-foreground mb-4 max-w-[640px] text-balance">
            It enables semantic retrieval across heterogeneous satellite sensors
            — SAR (Sentinel-1), Multispectral (Sentinel-2), and Gaofen-1 PAN/MS
            — in a single unified embedding space, achieving sub-30ms end-to-end
            query latency.
          </p>
        </>
      ),
    },
    {
      question: "How does the cross-modal retrieval work under the hood?",
      answer: (
        <>
          <p className="text-muted-foreground mb-4 max-w-[600px]">
            SABER uses a DOFA (Domain-Oriented Foundation Architecture)
            ViT-Base encoder with wavelength-conditioned hypernetworks to
            dynamically adapt patch projections to any sensor modality.
          </p>
          <p className="text-muted-foreground mb-4 max-w-[600px]">
            A Conditional Flow Matching (CFM) ODE latent bridge then translates
            query embeddings from one sensor manifold (e.g., SAR) to the target
            manifold (e.g., Optical), enabling geometric alignment without
            joint-sensor retraining.
          </p>
          <p className="text-muted-foreground mb-4 max-w-[600px]">
            The combined FAISS IndexFlatIP vector search completes retrieval
            in under 1ms for a 10,000-item gallery.
          </p>
        </>
      ),
    },
    {
      question: "What dataset and evaluation protocol were used?",
      answer: (
        <>
          <p className="text-muted-foreground mb-4 max-w-[580px]">
            SABER was trained and evaluated on BEN-14K — a 14,832-sample
            paired Sentinel-1 SAR / Sentinel-2 Multispectral dataset with
            19-class BigEarthNet multi-hot land cover labels.
          </p>
          <p className="text-muted-foreground mb-4 max-w-[580px]">
            Evaluation uses a strict 80% Gallery / 20% Query partition with
            100% non-synthetic real satellite data. Metrics reported include
            F1@5, F1@10, mAP@5, Precision@5, and Recall@5 across both
            cross-modal and same-modal directions.
          </p>
        </>
      ),
    },
    {
      question: "How parameter-efficient is SABER compared to baselines?",
      answer: (
        <>
          <p className="text-muted-foreground mb-4 max-w-[580px]">
            SABER uses LoRA adapters (rank r=16, α=32) on the attention and MLP
            layers of the frozen DOFA ViT backbone. Only <strong>294.9K
            parameters out of 111.6M (0.26%)</strong> are trainable during
            encoder training — the rest remain completely frozen.
          </p>
          <p className="text-muted-foreground mb-4 max-w-[580px]">
            This enables training on a single budget GPU with under 1GB VRAM,
            compared to 100% full fine-tuning required by baselines like
            RemoteCLIP (149M params) and X-JEPA (86M params).
          </p>
        </>
      ),
    },
    {
      question: "Is SABER open-source and how can I run it?",
      answer: (
        <>
          <p className="text-muted-foreground mb-4 max-w-[580px]">
            Yes — SABER is fully open-source under the MIT License. You can
            clone the repository, install dependencies, and run the live
            retrieval dashboard locally.
          </p>
          <p className="text-muted-foreground mb-4 max-w-[580px]">
            See the{" "}
            <Link
              href={`${siteConfig.links.github}/blob/master/docs/TRAINING_DOCUMENTATION.md`}
              className="text-foreground underline"
            >
              Training Documentation
            </Link>{" "}
            for a step-by-step guide to reproduce results and run training.
          </p>
        </>
      ),
    },
    {
      question: "What are the BEN-14K SOTA benchmark results?",
      answer: (
        <>
          <p className="text-muted-foreground mb-4 max-w-[580px]">
            On the 80/20 BEN-14K real data partition:
          </p>
          <ul className="text-muted-foreground list-disc pl-5 mb-4 max-w-[580px] space-y-1 text-sm">
            <li><strong>S1 → S2 Cross-Modal F1@5:</strong> 73.51%</li>
            <li><strong>S2 → S1 Cross-Modal F1@5:</strong> 73.10%</li>
            <li><strong>S1 → S1 Same-Modal F1@5:</strong> 75.40%</li>
            <li><strong>S2 → S2 Same-Modal F1@5:</strong> 76.38%</li>
            <li><strong>Cross-Modal mAP@5:</strong> 91.49%</li>
            <li><strong>Average Query Latency:</strong> 28.48 ms (0.97ms FAISS)</li>
          </ul>
          <p className="text-muted-foreground mb-4 max-w-[580px]">
            Full peer comparison against MAE, SatMAE, RemoteCLIP, X-JEPA, and
            CR-JEPA is available on the{" "}
            <Link href="/dashboard/format/training" className="text-foreground underline">
              Training Dashboard
            </Link>.
          </p>
        </>
      ),
    },
  ],
  className,
}: FAQProps) {
  return (
    <Section className={className}>
      <div className="max-w-container mx-auto flex flex-col items-center gap-8">
        <h2 className="text-center text-3xl font-semibold sm:text-5xl">
          {title}
        </h2>
        {items !== false && items.length > 0 && (
          <Accordion type="single" collapsible className="w-full max-w-[800px]">
            {items.map((item, index) => (
              <AccordionItem
                key={item.value ?? item.question}
                value={item.value || `item-${index + 1}`}
              >
                <AccordionTrigger>{item.question}</AccordionTrigger>
                <AccordionContent>{item.answer}</AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        )}
      </div>
    </Section>
  );
}
