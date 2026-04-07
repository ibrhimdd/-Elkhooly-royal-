import { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, Modality, GenerateContentResponse } from "@google/genai";
import { Camera, RefreshCw, Volume2, VolumeX, Loader2, History, Sparkles, X, Maximize2, RotateCcw, ArrowRight } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import ReactMarkdown from 'react-markdown';

// --- Constants & Types ---
const MODEL_NAME = "gemini-3-flash-preview"; // Stable model for text/vision
const TTS_MODEL_NAME = "gemini-2.5-flash-preview-tts"; // Dedicated TTS model
const SYSTEM_INSTRUCTION = "أنت مرشد متحف ملكي قديم. تعرف على الأثر في الصورة واشرح أهميته التاريخية بأسلوب ملكي مهيب ومختصر جداً (أقل من 50 كلمة)، ثم اختم بسؤال تحفيزي.";

interface Message {
  role: 'user' | 'assistant';
  text: string;
  image?: string;
  audio?: string;
  isStreaming?: boolean;
}

export default function App() {
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isMuted, setIsMuted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'camera' | 'history'>('camera');

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // --- Camera Setup ---
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          facingMode: 'environment',
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          setIsCameraReady(true);
          setError(null);
        };
      }
    } catch (err) {
      console.error("Error accessing camera:", err);
      setError("تعذر الوصول إلى الكاميرا. يرجى التأكد من منح الأذونات اللازمة.");
    }
  }, []);

  useEffect(() => {
    startCamera();
    return () => {
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach(track => track.stop());
      }
    };
  }, [startCamera]);

  useEffect(() => {
    const interval = setInterval(() => {
      if (!isCameraReady && videoRef.current && videoRef.current.videoWidth > 0) {
        setIsCameraReady(true);
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [isCameraReady]);

  // Scroll to top only when a NEW message starts
  useEffect(() => {
    if (scrollRef.current && messages.length > 0) {
      scrollRef.current.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }, [messages.length]);

  // --- AI Logic ---
  const processImage = async () => {
    if (!videoRef.current || !canvasRef.current || isProcessing) return;

    // Ensure video has dimensions
    if (!videoRef.current || videoRef.current.videoWidth === 0 || videoRef.current.videoHeight === 0) {
      setError("الكاميرا ليست جاهزة بعد. يرجى الانتظار ثانية حتى يكتمل التحميل.");
      // Force a re-check of camera readiness
      if (videoRef.current?.videoWidth && videoRef.current?.videoHeight) {
        setIsCameraReady(true);
      }
      return;
    }

    setIsProcessing(true);
    setError(null);
    
    // Auto-switch to results view immediately
    setActiveTab('history');

    // Create a temporary message for streaming
    const tempMessage: Message = {
      role: 'assistant',
      text: '',
      isStreaming: true
    };
    setMessages(prev => [tempMessage, ...prev]);

    try {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      
      const targetWidth = 640;
      const scaleFactor = targetWidth / video.videoWidth;
      canvas.width = targetWidth;
      canvas.height = video.videoHeight * scaleFactor;
      
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error("Could not get canvas context");

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const base64Image = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
      
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      
      const streamResponse = await ai.models.generateContentStream({
        model: MODEL_NAME,
        contents: {
          parts: [
            { text: SYSTEM_INSTRUCTION },
            { inlineData: { data: base64Image, mimeType: 'image/jpeg' } }
          ]
        }
      });

      let fullText = "";
      let audioTriggered = false;
      for await (const chunk of streamResponse) {
        const textChunk = (chunk as GenerateContentResponse).text || "";
        fullText += textChunk;
        
        // Trigger audio early (after 50 chars) to overlap with text streaming
        if (!audioTriggered && fullText.length > 50 && !isMuted) {
          audioTriggered = true;
          generateAudio(fullText);
        }

        setMessages(prev => {
          const newMessages = [...prev];
          if (newMessages[0]) {
            newMessages[0] = { ...newMessages[0], text: fullText };
          }
          return newMessages;
        });
      }

      // Finalize the message state
      setMessages(prev => {
        const newMessages = [...prev];
        if (newMessages[0]) {
          newMessages[0] = { 
            ...newMessages[0], 
            text: fullText, 
            isStreaming: false,
            image: `data:image/jpeg;base64,${base64Image}`
          };
        }
        return newMessages;
      });

      // If text was too short to trigger early, trigger it now
      if (!isMuted && fullText && !audioTriggered) {
        generateAudio(fullText);
      }

    } catch (err) {
      console.error("Processing Error:", err);
      const errorMsg = err instanceof Error ? err.message : JSON.stringify(err);
      setError(`حدث خطأ: ${errorMsg.includes('INVALID_ARGUMENT') ? 'فشل في معالجة الصورة' : 'يرجى المحاولة مرة أخرى'}`);
      setMessages(prev => prev.slice(1));
      setActiveTab('camera');
    } finally {
      setIsProcessing(false);
    }
  };

  const generateAudio = async (text: string) => {
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      
      const ttsResponse = await ai.models.generateContent({
        model: TTS_MODEL_NAME,
        contents: [{ parts: [{ text: `تحدث بصوت ملكي مهيب ومختصر: ${text}` }] }],
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: 'Kore' },
            },
          },
        },
      });

      const base64Audio = ttsResponse.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      if (base64Audio) {
        const binary = atob(base64Audio);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        const audioUrl = createWavUrl(bytes, 24000);
        
        setMessages(prev => {
          const newMessages = [...prev];
          if (newMessages[0]) {
            newMessages[0] = { ...newMessages[0], audio: audioUrl };
          }
          return newMessages;
        });

        const audio = new Audio(audioUrl);
        audio.play().catch(e => console.error("Audio playback failed:", e));
      }
    } catch (ttsErr) {
      console.error("TTS Error:", ttsErr);
    }
  };

  const createWavUrl = (pcmData: Uint8Array, sampleRate: number) => {
    const buffer = new ArrayBuffer(44 + pcmData.length);
    const view = new DataView(buffer);
    view.setUint32(0, 0x52494646, false);
    view.setUint32(4, 36 + pcmData.length, true);
    view.setUint32(8, 0x57415645, false);
    view.setUint32(12, 0x666d7420, false);
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    view.setUint32(36, 0x64617461, false);
    view.setUint32(40, pcmData.length, true);
    for (let i = 0; i < pcmData.length; i++) view.setUint8(44 + i, pcmData[i]);
    const blob = new Blob([buffer], { type: 'audio/wav' });
    return URL.createObjectURL(blob);
  };

  const resetSearch = () => {
    setActiveTab('camera');
  };

  return (
    <div className="fixed inset-0 bg-[#0a0a0a] text-[#f5f2ed] font-serif selection:bg-[#c5a059] selection:text-black flex flex-col overflow-hidden overscroll-behavior-none" dir="rtl">
      
      {/* --- Hieroglyphic Background Layer --- */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden opacity-5 select-none z-0">
        <div className="absolute inset-0 flex flex-wrap gap-12 p-12 text-6xl leading-relaxed text-[#c5a059]">
          {Array.from({ length: 100 }).map((_, i) => (
            <span key={i}>𓁹 𓆃 𓏏 𓅓 𓇋 𓈖 𓏏 𓅓 𓇋 𓈖 𓏏 𓅓 𓇋 𓈖 𓏏 𓅓 𓇋 𓈖 𓏏 𓅓 𓇋 𓈖</span>
          ))}
        </div>
      </div>

      {/* --- Background Atmosphere --- */}
      <div className="absolute inset-0 pointer-events-none opacity-30 z-0">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,#c5a05922_0%,transparent_70%)]" />
        <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/paper-fibers.png')] opacity-20" />
      </div>

      {/* --- Header --- */}
      <header className="relative z-50 flex justify-between items-center px-8 py-6 border-b border-[#c5a05922] bg-[#0a0a0a]/80 backdrop-blur-xl shrink-0">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 border border-[#c5a059] rounded-full flex items-center justify-center text-[#c5a059]">
            <Sparkles size={20} />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-[0.1em] text-[#c5a059] uppercase">AI Museum Guide</h1>
            <p className="text-[10px] text-[#c5a059]/60 uppercase tracking-[0.3em] font-sans">Royal Egyptian Protocol</p>
          </div>
        </div>
        
        <div className="flex items-center gap-6">
          <button 
            onClick={() => setIsMuted(!isMuted)}
            className={`transition-all duration-300 ${isMuted ? 'text-red-400/60' : 'text-[#c5a059]'}`}
          >
            {isMuted ? <VolumeX size={22} /> : <Volume2 size={22} />}
          </button>
          <button 
            onClick={() => setActiveTab(activeTab === 'camera' ? 'history' : 'camera')}
            className="text-[#c5a059] hover:scale-110 transition-transform"
          >
            {activeTab === 'camera' ? <History size={22} /> : <X size={22} />}
          </button>
        </div>
      </header>

      {/* --- Main Content --- */}
      <main className="relative flex-1 flex flex-col md:flex-row overflow-hidden z-10">
        
        {/* Camera Viewport */}
        <div className={`relative flex-1 transition-all duration-700 ease-in-out h-full ${activeTab === 'history' ? 'hidden md:block md:w-0 opacity-0 pointer-events-none' : 'w-full opacity-100'}`}>
          <div className="absolute inset-0 bg-black">
            <video 
              ref={videoRef} 
              autoPlay 
              playsInline 
              onLoadedMetadata={() => setIsCameraReady(true)}
              className={`w-full h-full object-cover transition-opacity duration-1000 ${isCameraReady ? 'opacity-80' : 'opacity-0'}`}
            />
            
            {isProcessing && (
              <motion.div 
                initial={{ top: '0%' }}
                animate={{ top: '100%' }}
                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                className="absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-[#c5a059] to-transparent shadow-[0_0_20px_#c5a059] z-10"
              />
            )}

            <div className="absolute inset-12 pointer-events-none border border-[#c5a05922]">
              <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-[#c5a059]" />
              <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-[#c5a059]" />
              <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-[#c5a059]" />
              <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-[#c5a059]" />
            </div>
          </div>

          <div className="absolute bottom-12 left-0 right-0 flex flex-col items-center gap-6 z-20">
            <AnimatePresence>
              {error && (
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 20 }}
                  className="bg-red-900/40 border border-red-500/50 backdrop-blur-md px-6 py-3 rounded-full text-red-200 text-sm flex items-center gap-3"
                >
                  {error}
                  <button onClick={startCamera} className="underline font-bold">إعادة تشغيل</button>
                </motion.div>
              )}
            </AnimatePresence>

            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={processImage}
              disabled={isProcessing || !isCameraReady}
              className={`
                relative w-24 h-24 rounded-full flex items-center justify-center transition-all duration-500
                ${isProcessing 
                  ? 'bg-[#c5a05922] border-2 border-[#c5a059] animate-pulse' 
                  : 'bg-transparent border-4 border-[#c5a059] hover:bg-[#c5a05911]'
                }
              `}
            >
              <div className={`w-16 h-16 rounded-full ${isProcessing ? 'bg-[#c5a059]' : 'bg-white/10'} flex items-center justify-center transition-colors`}>
                {isProcessing ? <Loader2 className="animate-spin text-black" size={32} /> : <Camera className="text-[#c5a059]" size={32} />}
              </div>
              
              <span className="absolute -bottom-10 whitespace-nowrap text-[10px] uppercase tracking-[0.4em] font-sans font-bold text-[#c5a059]">
                {isProcessing ? 'Analyzing...' : 'Analyze Artifact'}
              </span>
            </motion.button>
          </div>
        </div>

        {/* Results / History Sidebar */}
        <div className={`
          flex-1 flex flex-col bg-[#0f0f0f]/95 backdrop-blur-md border-r border-[#c5a05922] transition-all duration-700 ease-in-out h-full
          ${activeTab === 'camera' ? 'hidden md:flex md:w-[450px] translate-x-full md:translate-x-0' : 'w-full translate-x-0'}
        `}>
          <div className="p-8 border-b border-[#c5a05911] flex justify-between items-center shrink-0">
            <h2 className="text-lg font-bold tracking-widest text-[#c5a059] uppercase">Insights</h2>
            <button 
              onClick={resetSearch}
              className="flex items-center gap-2 text-[10px] text-[#c5a059] font-sans uppercase tracking-widest hover:opacity-70 transition-opacity"
            >
              <ArrowRight size={14} /> New Scan
            </button>
          </div>

          <div ref={scrollRef} className="flex-1 overflow-y-auto p-8 space-y-12 custom-scrollbar touch-pan-y overscroll-behavior-contain">
            <AnimatePresence initial={false}>
              {messages.length === 0 && (
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="h-full flex flex-col items-center justify-center text-center space-y-6 opacity-20"
                >
                  <Maximize2 size={48} />
                  <p className="text-sm italic tracking-widest">وجه الكاميرا نحو التاريخ...</p>
                </motion.div>
              )}

              {messages.map((msg, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="group relative"
                >
                  <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center gap-3">
                      <div className="w-2 h-2 rounded-full bg-[#c5a059]" />
                      <span className="text-[10px] uppercase tracking-[0.3em] font-sans font-bold text-[#c5a059]/60">Royal Archive</span>
                    </div>
                    {msg.audio && (
                      <button 
                        onClick={() => new Audio(msg.audio).play()}
                        className="text-[#c5a059] hover:scale-125 transition-transform"
                      >
                        <Volume2 size={16} />
                      </button>
                    )}
                  </div>

                  {msg.image && (
                    <div className="mb-6 rounded-2xl overflow-hidden border border-[#c5a05922] aspect-video">
                      <img src={msg.image} alt="Artifact" className="w-full h-full object-cover grayscale hover:grayscale-0 transition-all duration-700" />
                    </div>
                  )}

                  <div className={`prose prose-invert max-w-none text-[#f5f2ed]/90 leading-relaxed text-lg font-serif ${msg.isStreaming ? 'animate-pulse' : ''}`}>
                    <ReactMarkdown>{msg.text}</ReactMarkdown>
                  </div>

                  <div className="mt-12 h-px bg-gradient-to-r from-[#c5a05922] via-transparent to-transparent" />
                </motion.div>
              ))}
            </AnimatePresence>
          </div>

          {activeTab === 'history' && !isProcessing && (
            <div className="p-8 border-t border-[#c5a05911] bg-[#0a0a0a]/50 shrink-0">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={resetSearch}
                className="w-full py-4 bg-[#c5a059] text-black font-sans font-bold uppercase tracking-[0.2em] rounded-xl flex items-center justify-center gap-3 shadow-2xl"
              >
                <RotateCcw size={18} /> بحث جديد
              </motion.button>
            </div>
          )}
        </div>
      </main>

      <canvas ref={canvasRef} className="hidden" />

      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #c5a05922; border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #c5a05944; }
        
        /* Ensure scrolling works on mobile */
        .touch-pan-y {
          touch-action: pan-y;
          -webkit-overflow-scrolling: touch;
        }
      `}</style>
    </div>
  );
}
