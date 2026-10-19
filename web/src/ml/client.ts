import type { DecodeMode } from './meta';
import type { ModelInput } from './preprocess';
import type {
  Backend,
  DecoderSettings,
  Recognition,
  WorkerRequest,
  WorkerResponse,
} from './protocol';

/** Promise wrapper around the inference worker. One instance per loaded model. */
export class ModelClient {
  private readonly worker: Worker;
  private nextId = 1;
  private readonly pending = new Map<
    number,
    { resolve: (r: Recognition) => void; reject: (e: Error) => void }
  >();
  readonly ready: Promise<Backend>;

  constructor() {
    this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    this.ready = new Promise<Backend>((resolve, reject) => {
      this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
        const message = event.data;
        switch (message.type) {
          case 'ready':
            resolve(message.backend);
            break;
          case 'init-error':
            reject(new Error(message.message));
            break;
          case 'result':
            this.pending.get(message.id)?.resolve(message.result);
            this.pending.delete(message.id);
            break;
          case 'error':
            this.pending.get(message.id)?.reject(new Error(message.message));
            this.pending.delete(message.id);
            break;
        }
      };
      this.worker.onerror = (event) => {
        event.preventDefault();
        reject(new Error(event.message || 'the model worker failed to start'));
      };
    });
    this.send({ type: 'init' });
  }

  private send(request: WorkerRequest, transfer: Transferable[] = []) {
    this.worker.postMessage(request, transfer);
  }

  recognise(input: ModelInput, mode: DecodeMode, decoder: DecoderSettings): Promise<Recognition> {
    const id = this.nextId++;
    return new Promise<Recognition>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      const copy = input.data.slice();
      this.send({ type: 'recognise', id, input: copy, width: input.width, mode, decoder }, [
        copy.buffer,
      ]);
    });
  }

  dispose() {
    this.worker.terminate();
    for (const { reject } of this.pending.values()) reject(new Error('cancelled'));
    this.pending.clear();
  }
}
