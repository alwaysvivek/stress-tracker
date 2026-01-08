import React, { useEffect, useState, useRef } from 'react';
import { useMouseTracker, type MouseData } from '../hooks/useMouseTracker';

interface TrackerCanvasProps {
    onSessionComplete: (data: MouseData[]) => void;
}

export const TrackerCanvas: React.FC<TrackerCanvasProps> = ({ onSessionComplete }) => {
    const [isActive, setIsActive] = useState(false);
    const { getAndClearBuffer, resetTimer } = useMouseTracker(isActive);
    const [targets, setTargets] = useState<{ x: number, y: number, id: number }[]>([]);
    const [score, setScore] = useState(0);
    // Removed unused requestRef

    // Simple game loop or interval to spawn targets
    useEffect(() => {
        if (isActive) {
            // Spawn a target
            spawnTarget();
        }
    }, [isActive]);

    const spawnTarget = () => {
        const x = Math.random() * (window.innerWidth - 100) + 50;
        const y = Math.random() * (window.innerHeight - 100) + 50;
        setTargets([{ x, y, id: Date.now() }]);
    };

    const handleTargetClick = (id: number) => {
        setScore(s => s + 1);
        spawnTarget(); // Move to next
    };

    const startSession = () => {
        setIsActive(true);
        resetTimer();
        setScore(0);
        // Auto-end after 10 seconds for prototype
        setTimeout(() => {
            finishSession();
        }, 10000);
    };

    const finishSession = () => {
        setIsActive(false);
        const sessionData = getAndClearBuffer();
        console.log("Session complete, data points:", sessionData.length);
        onSessionComplete(sessionData); // Send to parent/backend
    };

    return (
        <div className="relative w-full h-screen bg-slate-900 text-white overflow-hidden">
            {!isActive && (
                <div className="absolute inset-0 flex flex-col items-center justify-center z-10 bg-black/50">
                    <h1 className="text-4xl font-bold mb-4">Stress Mouse Tracker</h1>
                    <button
                        onClick={startSession}
                        className="px-6 py-3 bg-blue-600 hover:bg-blue-500 rounded-lg text-lg font-semibold transition"
                    >
                        Start Calibration Task (10s)
                    </button>
                    {score > 0 && <p className="mt-4 text-xl text-green-400">Previous Score: {score}</p>}
                </div>
            )}

            {isActive && targets.map(t => (
                <div
                    key={t.id}
                    className="absolute w-12 h-12 bg-red-500 rounded-full cursor-pointer hover:bg-red-400 transform transition-transform active:scale-95"
                    style={{ left: t.x, top: t.y }}
                    onClick={(e) => {
                        e.stopPropagation();
                        handleTargetClick(t.id);
                    }}
                />
            ))}

            <div className="absolute top-4 left-4 text-gray-400 pointer-events-none">
                Score: <span className="text-white">{score}</span>
            </div>
        </div>
    );
};
