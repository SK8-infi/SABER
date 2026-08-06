import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "../../ui/accordion";
import { Section } from "../../ui/section";

const FAQ_ITEMS = [
  {
    q: "What does SABER stand for?",
    a: "SABER stands for Sensor-Agnostic Bridged Embedding Retrieval. It is a cross-modal satellite image retrieval system that bridges SAR radar and optical multispectral imagery into a shared embedding space, enabling retrieval across sensors in under 30ms.",
  },
  {
    q: "Why is cross-modal retrieval hard?",
    a: "SAR (Synthetic Aperture Radar) and optical images of the same location look completely different. SAR measures radar backscatter — urban areas appear bright, water appears dark. Optical images show visual color. Standard embedding models trained on one modality fail to match across modalities because the embeddings cluster separately in latent space.",
  },
  {
    q: "What is the CFM Bridge and why does it matter?",
    a: "The Continuous Flow Matching (CFM) Bridge is SABER's core innovation. It trains a neural ODE that smoothly transports SAR embeddings from their cluster to the optical embedding hypersphere at inference time. Without it, cross-modal retrieval F1@5 drops from 83.4% to 72.1% — an 11-point gap. The bridge runs in ~12ms on CPU.",
  },
  {
    q: "What datasets does SABER support?",
    a: "SABER supports two datasets: BEN-14K (BigEarthNet v1 14K paired Sentinel-1 SAR + Sentinel-2 multispectral scenes, 19 land-cover classes) and DSRSID (Gaofen-1 PAN/MS paired scenes). Both use real satellite imagery — no synthetic data at inference time.",
  },
  {
    q: "How fast is SABER?",
    a: "End-to-end query latency on CPU is 28.48ms: preprocessing (0.80ms) + DOFA ViT feature extraction (14.20ms) + CFM bridge ODE (12.51ms) + FAISS cosine search (0.97ms). On a GPU this drops to sub-10ms. The FAISS flat index searches 14,832 vectors in under 1ms.",
  },
  {
    q: "What is DOFA ViT and why use it as the backbone?",
    a: "DOFA (Dynamic One-For-All) is a Vision Transformer pretrained across 100+ Earth observation bands using wavelength conditioning. Instead of fixed RGB channels, DOFA dynamically adapts to any number of input channels by conditioning on the physical wavelength of each band. This makes it uniquely suited for multi-sensor Earth observation tasks.",
  },
  {
    q: "How does SABER handle different numbers of input channels?",
    a: "SABER uses a lightweight input adapter (1×1 conv) that maps any number of input channels (2 for S1 SAR, 12 for S2 optical, 1 for PAN, 4 for MS) to the 768-dim ViT embedding space. Only 294.9K parameters (0.26% of the 111.6M total) are trainable — everything else is frozen from DOFA pretraining.",
  },
  {
    q: "What is ISRO Problem Statement 11?",
    a: "PS-11 from ISRO's Breakthrough Aerospace Hackathon (BAH) 2026 requires building a cross-modal satellite image retrieval system that achieves F1@5 > 75% on BEN-14K cross-modal queries with retrieval latency under 100ms. SABER achieves 83.4% F1@5 at 28ms — well beyond both thresholds.",
  },
];

export default function FAQ() {
  return (
    <Section id="faq">
      <div className="max-w-container mx-auto flex flex-col items-center gap-10">
        <div className="flex flex-col items-center gap-3 text-center">
          <span className="text-xs font-bold uppercase tracking-widest text-[#FBBA72]">FAQ</span>
          <h2 className="text-3xl font-semibold sm:text-5xl">Common Questions</h2>
          <p className="text-muted-foreground max-w-[520px] text-base">
            Everything you need to understand how SABER works and why it matters for Earth observation.
          </p>
        </div>
        <Accordion type="single" collapsible className="w-full max-w-[800px]">
          {FAQ_ITEMS.map((item, i) => (
            <AccordionItem key={i} value={`item-${i}`}>
              <AccordionTrigger className="text-left">{item.q}</AccordionTrigger>
              <AccordionContent>
                <p className="text-muted-foreground leading-relaxed max-w-[680px]">{item.a}</p>
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </div>
    </Section>
  );
}
