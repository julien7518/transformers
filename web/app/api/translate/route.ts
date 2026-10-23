import {proxy} from '@/lib/proxy';
import {mockTranslate} from '@/lib/mock';
export const runtime = 'nodejs';
export async function POST(request: Request) {
  return process.env.ATTENTION_MOCK === 'true' ? mockTranslate(request) : proxy('translate', request);
}
