import React, { useState, useEffect } from 'react';
import { useKeystrokeTracker, type RawKeyData } from '../hooks/useKeystrokeTracker';

interface TypingTestProps {
    onSessionComplete: (data: RawKeyData[]) => void;
}

const TARGET_TEXT = "The quick brown fox jumps over the lazy dog. Programming is the art of telling another human what one wants the computer to do.";

export const TypingTest: React.FC<TypingTestProps> = ({ onSessionComplete }) => {
    const [isActive, setIsActive] = useState(false);
    const [inputValue, setInputValue] = useState("");
    const { getAndClearKeystrokes, resetKeyTimer } = useKeystrokeTracker(isActive);
    const [isComplete, setIsComplete] = useState(false);

    useEffect(() => {
        if (inputValue === TARGET_TEXT && isActive) {
            finishSession();
        }
    }, [inputValue, isActive]);

    const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        if (!isActive) {
            setIsActive(true);
            resetKeyTimer();
        }
        setInputValue(e.target.value);
    };

    const finishSession = () => {
        setIsActive(false);
        setIsComplete(true);
        const data = getAndClearKeystrokes();
        console.log("Typing session complete, keystrokes:", data.length);
        onSessionComplete(data);
    };

    const reset = () => {
        setInputValue("");
        setIsActive(false);
        setIsComplete(false);
        resetKeyTimer();
    };

    return (
        <div className="relative w-full max-w-2xl mx-auto p-8 bg-white text-gray-800 rounded-xl shadow-lg mt-10">
            <h2 className="text-2xl font-bold mb-4">Typing Stress Test</h2>
            <p className="mb-4 text-gray-600">Type the following text as quickly and accurately as possible:</p>

            <div className="p-4 bg-gray-100 rounded-lg mb-6 text-lg font-mono leading-relaxed select-none">
                {TARGET_TEXT}
            </div>

            <textarea
                value={inputValue}
                onChange={handleChange}
                placeholder="Start typing here..."
                className="w-full h-32 p-4 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:ring focus:ring-blue-200 transition text-lg"
                disabled={isComplete}
                onPaste={(e) => e.preventDefault()}
            />

            {isComplete && (
                <div className="mt-4 p-4 bg-green-100 text-green-700 rounded-lg">
                    Test Complete! Data sent.
                    <button onClick={reset} className="ml-4 underline font-bold">Try Again</button>
                </div>
            )}

            <div className="mt-2 text-sm text-gray-500">
                {isActive ? "Tracking..." : "Ready"} | Length: {inputValue.length}/{TARGET_TEXT.length}
            </div>
        </div>
    );
};
