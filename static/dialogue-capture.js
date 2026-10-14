// Continuous area resampling: packet boundaries do not reset the sample clock.
class DialogueCapture extends AudioWorkletProcessor {
  constructor() {
    super(); this.width = sampleRate / 16000; this.remaining = this.width;
    this.integral = 0; this.samples = []; this.energy = 0; this.count = 0; this.active = true;
    this.port.onmessage = event => {
      if (event.data === 'flush') {
        this.active = false; this.emit(); this.port.postMessage({type:'flushed'});
      }
    };
  }
  emit() {
    if (!this.samples.length) return;
    const buffer = new ArrayBuffer(this.samples.length * 2), view = new DataView(buffer);
    this.samples.forEach((value, i) => view.setInt16(i*2, value, true));
    this.port.postMessage({type:'pcm', buffer, rms:Math.sqrt(this.energy/Math.max(1,this.count))}, [buffer]);
    this.samples = []; this.energy = 0; this.count = 0;
  }
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!this.active || !channel) return true;
    for (const input of channel) {
      const x = Number.isFinite(input) ? Math.max(-1,Math.min(1,input)) : 0;
      let available = 1;
      while (available > 1e-9) {
        const take = Math.min(available,this.remaining);
        this.integral += x*take; this.remaining -= take; available -= take;
        if (this.remaining < 1e-9) {
          const value = this.integral/this.width;
          this.samples.push(Math.round(value*(value < 0 ? 32768 : 32767)));
          this.energy += value*value; this.count++;
          this.remaining = this.width; this.integral = 0;
          if (this.samples.length === 1600) this.emit();
        }
      }
    }
    return true;
  }
}
registerProcessor('dialogue-capture', DialogueCapture);
