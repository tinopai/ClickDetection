const express = require('express');
const fs = require('fs');
const app = express();

app.post('/saveClicks', express.json(), (req, res)=>{
    const clicks = req.body?.clicks || [0]
    const total = req.body.total
    const type = req.body.type

    fs.writeFileSync(`clicks/${type}/${total}_${Date.now()}.txt`, `${JSON.stringify(clicks)}`);
    res.status(200).json({
        success: true
    })
})

app.get('/', (req, res)=>{
    res.sendFile(__dirname + '/cpsScan.html');
})

app.listen(3000, ()=>{
    console.log(
        `Server is running on port 3000`
    )
});