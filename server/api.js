const express = require('express');
const fs = require('fs');
const app = express();

const allowedTypes = ['human', 'complex', 'simple']
app.post('/saveClicks', express.json(), (req, res)=>{
    try {
        const clicks = req.body?.clicks || [0]
        const total = req.body?.total
        const type = req.body?.type

        if (!allowedTypes.includes(type) || !(total.match(/^[0-9]{0,10}$/)))
            return res.status(500).json({
                success: false,
                message: 'Please don\'t troll, you wont achieve anything'
            })
        

        fs.writeFileSync(`clicks/${type}/${total}_${Date.now()}.txt`, `${JSON.stringify(clicks)}`);
        res.status(200).json({
            success: true
        })
    } catch(ex) {
        res.status(500).json({
            success: false,
            message: 'Could not save',
        })
    }
})

app.get('/', (req, res)=>{
    res.sendFile(__dirname + '/cpsScan.html');
})

let port = process.env.PORT || 9171;
app.listen(port, ()=>{
    console.log(
        `Server is running on port ${port}`
    )
});