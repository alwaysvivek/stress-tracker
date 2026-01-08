import { useState } from 'react';
import { TrackerCanvas } from './components/TrackerCanvas';
import { TypingTest } from './components/TypingTest';
import type { MouseData } from './hooks/useMouseTracker';
import type { RawKeyData } from './hooks/useKeystrokeTracker';

type TaskType = 'mouse' | 'typing';

function App() {
  const [status, setStatus] = useState<string>('Ready');
  const [currentTask, setCurrentTask] = useState<TaskType | null>(null);

  const handleMouseSessionComplete = async (data: MouseData[]) => {
    await submitData({ movements: data, keystrokes: [] });
  };

  const handleTypingSessionComplete = async (data: RawKeyData[]) => {
    await submitData({ movements: [], keystrokes: data });
  };

  const submitData = async (payload: { movements: MouseData[], keystrokes: RawKeyData[] }) => {
    setStatus('Sending data...');
    try {
      const response = await fetch('http://localhost:8000/submit-session', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      const result = await response.json();
      setStatus(`Saved! Stress Score: ${result.stress_score.toFixed(2)}`);
      // Return to menu after short delay
      setTimeout(() => {
        setStatus('Ready');
        setCurrentTask(null);
      }, 3000);
    } catch (error) {
      console.error('Error sending session:', error);
      setStatus('Error saving data');
    }
  };

  return (
    <>
      <div className="absolute top-4 right-4 z-50 text-white bg-gray-800 px-4 py-2 rounded font-mono text-sm opacity-80 pointer-events-none">
        Status: {status}
      </div>

      {!currentTask && (
        <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white gap-6">
          <h1 className="text-4xl font-bold mb-8">Stress Detector Tasks</h1>
          <div className="flex gap-6">
            <button
              onClick={() => setCurrentTask('mouse')}
              className="w-64 h-40 bg-blue-600 hover:bg-blue-500 rounded-xl flex flex-col items-center justify-center transition transform hover:scale-105"
            >
              <span className="text-2xl font-bold">Mouse Tracking</span>
              <span className="text-blue-200 mt-2">Calibration / Stress Task</span>
            </button>
            <button
              onClick={() => setCurrentTask('typing')}
              className="w-64 h-40 bg-purple-600 hover:bg-purple-500 rounded-xl flex flex-col items-center justify-center transition transform hover:scale-105"
            >
              <span className="text-2xl font-bold">Typing Test</span>
              <span className="text-purple-200 mt-2">Keystroke Dynamics</span>
            </button>
          </div>
        </div>
      )}

      {currentTask === 'mouse' && (
        <div className="relative w-full h-full">
          <button
            onClick={() => setCurrentTask(null)}
            className="absolute top-4 left-4 z-50 text-white bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded"
          >
            &larr; Back
          </button>
          <TrackerCanvas onSessionComplete={handleMouseSessionComplete} />
        </div>
      )}

      {currentTask === 'typing' && (
        <div className="min-h-screen bg-gray-50 pt-20">
          <button
            onClick={() => setCurrentTask(null)}
            className="absolute top-4 left-4 z-50 text-white bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded"
          >
            &larr; Back
          </button>
          <TypingTest onSessionComplete={handleTypingSessionComplete} />
        </div>
      )}
    </>
  )
}

export default App
