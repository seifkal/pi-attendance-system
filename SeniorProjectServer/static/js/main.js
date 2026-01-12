let allSessions = [];
let currentSessionId = null;

// --- 1. Fetch List of Classes ---
async function fetchSessions() {
    try {
        const res = await fetch('/api/sessions');
        if (!res.ok) throw new Error("Failed to load session list");
        allSessions = await res.json();
        renderSidebar();
    } catch (e) { 
        console.error("Sidebar Error:", e); 
    }
}

// --- 2. Render Sidebar ---
function renderSidebar() {
    const list = document.getElementById('classList');
    const searchInput = document.getElementById('searchInput');
    
    if (!list || !searchInput) return;

    const term = searchInput.value.toLowerCase();
    const filtered = allSessions.filter(s => s.name.toLowerCase().includes(term) || s.date_str.includes(term));

    list.innerHTML = '';
    filtered.forEach(s => {
        const div = document.createElement('div');
        div.className = `class-item ${s.id === currentSessionId ? 'active' : ''}`;
        div.onclick = () => loadSession(s.id, s.name, s.date_str);
        div.innerHTML = `<div class="class-name">${s.name}</div><div class="class-date">${s.date_str}</div>`;
        list.appendChild(div);
    });
}

// --- 3. Load Specific Session ---
async function loadSession(id, name, date) {
    try {
        currentSessionId = id;
        
        // Update Header
        document.getElementById('sessionTitle').innerText = name;
        document.getElementById('sessionDate').innerText = `Report Date: ${date}`;
        
        // Show Delete Button
        const deleteBtn = document.getElementById('deleteBtn');
        if (deleteBtn) deleteBtn.style.display = 'flex';
        
        renderSidebar();

        const res = await fetch(`/api/session/${id}`);
        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        
        const students = await res.json();
        
        // --- STATISTICS LOGIC ---
        const presentStudents = students.filter(s => s.status !== 'Absent');
        const total = presentStudents.length;
        
        const avg = total > 0 
            ? presentStudents.reduce((acc, s) => acc + s.attention_pct, 0) / total 
            : 0;
        
        const alerts = students.filter(s => s.low_attention).length;

        document.getElementById('statPresent').innerText = total;
        document.getElementById('statAvgFocus').innerText = Math.round(avg) + "%";
        document.getElementById('statAlerts').innerText = alerts;
        
        // --- TABLE GENERATION ---
        const tbody = document.getElementById('studentTable');
        tbody.innerHTML = '';

        if (students.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="empty-state">No data recorded.</td></tr>';
            return;
        }

        students.forEach(s => {
            const attnColor = s.low_attention ? 'var(--danger)' : 'var(--success)';
            
            // --- BADGE STYLING LOGIC ---
            // If Absent: Red Background/Text. If Present: Green Background/Text.
            const isAbsent = s.status === 'Absent';
            const badgeStyle = isAbsent 
                ? 'background: rgba(239, 68, 68, 0.15); color: var(--danger);' 
                : 'background: rgba(16, 185, 129, 0.15); color: var(--success);';

            const row = document.createElement('tr');
            row.innerHTML = `
                <td style="font-weight:600; color:white;">${s.name}</td>
                
                <td>
                    <span class="badge" style="${badgeStyle}">${s.status}</span>
                </td>
                
                <td>${s.first_seen}</td>
                <td style="color:var(--text-muted);">${s.checks}</td>
                <td>
                    <div class="attn-wrapper">
                        <div class="attn-track">
                            <div class="attn-fill" style="width:${s.attention_pct}%; background:${attnColor};"></div>
                        </div>
                        <div class="attn-val" style="color:${attnColor}">${s.attention_pct}%</div>
                    </div>
                </td>`;
            tbody.appendChild(row);
        });

    } catch (error) {
        console.error(error);
        alert("Error loading report: " + error.message);
    }
}

// --- 4. Delete Function ---
async function deleteReport() {
    if (!currentSessionId) return;
    
    if (confirm("Are you sure you want to delete this report? This cannot be undone.")) {
        try {
            const res = await fetch(`/api/session/${currentSessionId}`, { method: 'DELETE' });
            if (res.ok) {
                alert("Report Deleted!");
                location.reload(); 
            } else {
                alert("Error deleting report.");
            }
        } catch (e) {
            alert("Connection error while deleting.");
        }
    }
}

// --- Initialization ---
const searchInput = document.getElementById('searchInput');
if (searchInput) {
    searchInput.addEventListener('keyup', renderSidebar);
}

fetchSessions();
setInterval(fetchSessions, 5000);