import { useEffect, useRef, useCallback } from 'react';

export interface KeystrokeData {
    key: string;
    keyDownTimestamp: number;
    keyUpTimestamp: number;
    dwellTime: number; // Duration held down
    flightTime: number; // Time since previous key up
}

// Simplified raw event tracking
export interface RawKeyData {
    key: string;
    action: 'down' | 'up';
    timestamp: number;
}

export const useKeystrokeTracker = (isActive: boolean) => {
    const bufferRef = useRef<RawKeyData[]>([]);
    const startTimeRef = useRef<number>(Date.now());

    useEffect(() => {
        if (!isActive) return;

        const handleKeyDown = (e: KeyboardEvent) => {
            // Avoid auto-repeat for hold? Or track it?
            if (e.repeat) return;
            bufferRef.current.push({
                key: e.key,
                action: 'down',
                timestamp: Date.now() - startTimeRef.current
            });
        };

        const handleKeyUp = (e: KeyboardEvent) => {
            bufferRef.current.push({
                key: e.key,
                action: 'up',
                timestamp: Date.now() - startTimeRef.current
            });
        };

        window.addEventListener('keydown', handleKeyDown);
        window.addEventListener('keyup', handleKeyUp);

        return () => {
            window.removeEventListener('keydown', handleKeyDown);
            window.removeEventListener('keyup', handleKeyUp);
        };
    }, [isActive]);

    const getAndClearKeystrokes = useCallback(() => {
        const data = [...bufferRef.current];
        bufferRef.current = [];
        return data;
    }, []);

    const resetKeyTimer = useCallback(() => {
        startTimeRef.current = Date.now();
        bufferRef.current = [];
    }, []);

    return { getAndClearKeystrokes, resetKeyTimer };
};
