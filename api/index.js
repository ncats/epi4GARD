// get search bar
const searchInput = document.getElementById("searchInput");

const namesFromDOM = document.getElementsByClassName("name");

// user inputs
searchInput.addEventListener("keyup", (event) => {
    const { value } = event.target;
    
    for (const nameElement of namesFromDOM) {
        
        // check current name to search input
        if (nameElement.includes(searchQuery)) {
            // if yes, display it
            nameElement.style.display = "block";
        } else {
            // if no, nothing to show
            nameElement.style.display = "none";
        }
    }
});