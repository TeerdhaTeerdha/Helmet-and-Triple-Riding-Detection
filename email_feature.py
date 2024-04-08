import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email import encoders
from os.path import basename

FROM_EMAIL = "sruthijvinu7@gmail.com"
FROM_EMAIL_PASSWORD = "tvignijcyteyvlko"

def send_email(recipient_email, subject, message, files):
    msg = MIMEMultipart()
    for f in files or []:
        with open(f, "rb") as fil:
            part = MIMEApplication(
                fil.read(),
                Name=basename(f)
            )
        
        part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
        msg.attach(part)
    
    
    msg['From'] = FROM_EMAIL
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(FROM_EMAIL, FROM_EMAIL_PASSWORD)
            server.sendmail(FROM_EMAIL, recipient_email, msg.as_string())
        print("Email sent successfully.")
    except Exception as e:
        print(f"Error: {e}")


# sender_email = 'your_email@gmail.com'
# sender_password = 'your_password'
# recipient_email = 'recipient_email@example.com'
# subject = 'Test Email'
# message = 'This is a test email.'
# send_email('warrior.prince652002@gmail.com', subject, message)


