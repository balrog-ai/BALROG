function updateTableByRepoHelper(data) {
    const tableBody = document.querySelector('#table-by-repo tbody');
    tableBody.innerHTML = ''; // Clear existing rows

    const sortedEntries = Object.entries(data).sort((a, b) => {
        const repoA = a[0].toLowerCase();
        const repoB = b[0].toLowerCase();
        if (repoA < repoB) return -1;
        if (repoA > repoB) return 1;
        return 0;
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

function updateTableByRepo(split, model) {
    const url = `https://raw.githubusercontent.com/swe-bench/experiments/main/evaluation/${split}/${model}/results/resolved_by_repo.json`;
    fetch(url)
        .then(response => response.json())
        .then(data => {
            updateTableByRepoHelper(data);
        })
        .catch(error => {
            console.error('Error fetching the JSON data:', error);
        });
}
