function updateTableByYearHelper(data) {
    const tableBody = document.querySelector('#table-by-year tbody');
    tableBody.innerHTML = ''; // Clear existing rows

    const sortedEntries = Object.entries(data).sort((a, b) => {
        const repoA = a[0].toLowerCase();
        const repoB = b[0].toLowerCase();

        const canBeIntegerA = Number.isInteger(parseInt(repoA, 10));
        const canBeIntegerB = Number.isInteger(parseInt(repoB, 10));

        if (canBeIntegerA && !canBeIntegerB) return -1;
        if (!canBeIntegerA && canBeIntegerB) return 1;
        if (canBeIntegerA && canBeIntegerB) return parseInt(repoB, 10) - parseInt(repoA, 10);
        return repoA.localeCompare(repoB);
    });

    for (const [repo, values] of sortedEntries) {
        const row = document.createElement('tr');
        
        const repoCell = document.createElement('td');
        repoCell.textContent = repo;
        row.appendChild(repoCell);
        
        const resolvedCell = document.createElement('td');
        resolvedCell.textContent = values.resolved;
        row.appendChild(resolvedCell);
        
        const totalCell = document.createElement('td');
        totalCell.textContent = values.total;
        row.appendChild(totalCell);

        const percentageCell = document.createElement('td');
        percentageCell.textContent = `${((values.resolved / values.total) * 100).toFixed(2)}%`;
        row.appendChild(percentageCell);
        
        tableBody.appendChild(row);
    }
}

function updateTableByYear(split, model) {
    const url = `https://raw.githubusercontent.com/swe-bench/experiments/main/evaluation/${split}/${model}/results/resolved_by_time.json`;
    fetch(url)
        .then(response => response.json())
        .then(data => {
            updateTableByYearHelper(data);
        })
        .catch(error => {
            console.error('Error fetching the JSON data:', error);
        });
}
