export const getGraphData = async (token: string) => {
    const res = await fetch('/api/graph/data', {
        headers: { Authorization: `Bearer ${token}` }
    });
    return await res.json();
};

export const searchGraph = async (token: string, query: string) => {
    const res = await fetch(`/api/graph/search?query=${encodeURIComponent(query)}`, {
        headers: { Authorization: `Bearer ${token}` }
    });
    return await res.json();
};
