css = '''
<style> 
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    background-color: #f8f9fa;
}
.chat-message.user {
    background-color: #e9ecef;
    border-left: 5px solid #1e3d59;
}
.chat-message.bot {
    background-color: #e9ecef;
    border-left: 5px solid #17a2b8;
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .avatar img {
    max-width: 68px;
    max-height: 68px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #212529;
}
'''

logo = '''
<div style="margin-bottom: 15px; text-align: center;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/a/a1/Escudo_Polic%C3%ADa_Nacional_de_Colombia.jpg" alt="Logo Policía Nacional" style="max-width: 25%; height: auto;">
</div>
'''