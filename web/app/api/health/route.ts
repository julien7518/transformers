import {proxy} from '@/lib/proxy';
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export async function GET() {
  return process.env.ATTENTION_MOCK === 'true' ? Response.json({status: 'ready', model_id: 'fixture-demo', limits: {max_input_chars: 500, max_source_tokens: 64, max_new_tokens: 63}}) : proxy('health');
}
