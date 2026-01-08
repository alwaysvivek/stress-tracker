import { useEffect, useCallback, useRef } from 'react';

export interface MouseData {
    x: number;
    y: number;
    timestamp: number;
    eventType: 'move' | 'click';
}

export const useMouseTracker = (isActive: boolean) => {
    // Data is in bufferRef to avoid re-renders on every move
    const bufferRef = useRef<MouseData[]>([]);
    const startTimeRef = useRef<number>(Date.now());

    useEffect(() => {
        if (!isActive) return;

        const handleMove = (e: MouseEvent) => {
            const point: MouseData = {
                x: e.clientX,
                y: e.clientY,
                timestamp: Date.now() - startTimeRef.current,
                eventType: 'move'
            };
            bufferRef.current.push(point);
        };

        const handleClick = (e: MouseEvent) => {
            const point: MouseData = {
                x: e.clientX,
                y: e.clientY,
                timestamp: Date.now() - startTimeRef.current,
                eventType: 'click'
            };
            bufferRef.current.push(point);
        };

        window.addEventListener('mousemove', handleMove);
        window.addEventListener('click', handleClick);

        return () => {
            window.removeEventListener('mousemove', handleMove);
            window.removeEventListener('click', handleClick);
        };
    }, [isActive]);

    const getAndClearBuffer = useCallback(() => {
        const data = [...bufferRef.current];
        bufferRef.current = [];
        return data;
    }, []);

    const resetTimer = useCallback(() => {
        startTimeRef.current = Date.now();
        bufferRef.current = [];
    }, []);

    return { getAndClearBuffer, resetTimer };
};
