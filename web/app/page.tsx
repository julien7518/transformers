import Translator from "@/components/translator";
export const dynamic = "force-dynamic";
export default function Page() {
  return <Translator mock={process.env.ATTENTION_MOCK === "true"} />;
}
